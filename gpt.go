// Package main содержит учебный пример реализации упрощённого трансформера с полным
// (но упрощённым) backpropagation, многопоточностью (через goroutines и каналы), режимом
// ask/ответ, и сохранением модели по сигналу.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"log"
	"math"
	"math/rand"
	"os"
	"os/signal"
	"runtime"
	"sort"
	"strings"
	"syscall"
	"time"
)

// ====================================================================================
// 1. Базовые операции (векторные, матричные)
// ====================================================================================

// dot вычисляет скалярное произведение (dot product) двух векторов a и b одинаковой длины.
func dot(a, b []float64) float64 {
	// Проверяем, что длины совпадают. Если нет, аварийно завершаем программу (log.Fatalf).
	if len(a) != len(b) {
		log.Fatalf("dot: размер векторов не совпадает: %d vs %d", len(a), len(b))
	}
	s := 0.0
	// Суммируем поэлементно произведения соответствующих элементов.
	for i := range a {
		s += a[i] * b[i]
	}
	// Возвращаем скалярное произведение.
	return s
}

// addVec складывает два вектора a и b одинаковой длины и возвращает результат.
func addVec(a, b []float64) []float64 {
	if len(a) != len(b) {
		log.Fatalf("addVec: размер векторов не совпадает: %d vs %d", len(a), len(b))
	}
	out := make([]float64, len(a))
	// Поэлементно складываем.
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// scaleVec умножает вектор a на скаляр s.
func scaleVec(a []float64, s float64) []float64 {
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] * s
	}
	return out
}

// matVecMul умножает матрицу A размера (m×n) на вектор x длины n, результат будет вектор длины m.
func matVecMul(A [][]float64, x []float64) []float64 {
	m := len(A)     // число строк
	n := len(A[0])  // число столбцов
	if len(x) != n {
		log.Fatalf("matVecMul: размер матрицы %dx%d и длина вектора %d", m, n, len(x))
	}
	out := make([]float64, m)
	// Для каждой строки i матрицы A, вычисляем скалярное произведение с x.
	for i := 0; i < m; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += A[i][j] * x[j]
		}
		out[i] = s
	}
	return out
}

// copyVec создает копию вектора a.
func copyVec(a []float64) []float64 {
	out := make([]float64, len(a))
	copy(out, a)
	return out
}

// randMat создает матрицу rows×cols со случайными значениями в диапазоне ±scale.
func randMat(rows, cols int, scale float64) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = (rand.Float64()*2 - 1) * scale
		}
	}
	return mat
}

// randVec создает вектор длины n со случайными значениями в диапазоне ±scale.
func randVec(n int, scale float64) []float64 {
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		out[i] = (rand.Float64()*2 - 1) * scale
	}
	return out
}

// ====================================================================================
// 2. Активационные функции (ReLU), softmax
// ====================================================================================

// reluForward применяет ReLU: out[i] = max(0, x[i])
func reluForward(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		} else {
			out[i] = 0
		}
	}
	return out
}

// reluBackward считает производную ReLU: если x[i]>0, dx[i]=dout[i], иначе 0
func reluBackward(dout, x []float64) []float64 {
	dx := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			dx[i] = dout[i]
		} else {
			dx[i] = 0
		}
	}
	return dx
}

// softmaxForward вычисляет softmax распределение для вектора x.
func softmaxForward(x []float64) []float64 {
	maxVal := x[0]
	// Ищем максимум, чтобы избежать переполнения.
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	exps := make([]float64, len(x))
	sumExp := 0.0
	// Считаем e^(x[i] - maxVal) для численной стабильности.
	for i, v := range x {
		e := math.Exp(v - maxVal)
		exps[i] = e
		sumExp += e
	}
	if sumExp == 0 {
		sumExp = 1e-9
	}
	out := make([]float64, len(x))
	// Нормируем
	for i := range out {
		out[i] = exps[i] / sumExp
	}
	return out
}

// ====================================================================================
// 3. TransformerConfig - глобальные гиперпараметры
// ====================================================================================

// TransformerConfig задает основные параметры трансформера.
type TransformerConfig struct {
	VocabSize     int     // Количество уникальных символов (размер словаря)
	EmbedSize     int     // Размерность эмбеддингов
	BlockSize     int     // Максимальная длина последовательности (контекст)
	MLPDim        int     // Размер скрытого слоя в MLP
	LearningRate  float64 // Скорость обучения (SGD)
	NumIterations int     // Количество итераций обучения
	BatchSize     int     // Размер батча (не используется в данном примере)
}

// ====================================================================================
// 4. Linear, EmbeddingLayer, LayerNorm, SelfAttention, MLPBlock
// ====================================================================================

// 4.1. Linear: один линейный слой W*x + B
type Linear struct {
	InDim  int
	OutDim int
	W      [][]float64
	B      []float64
	X      []float64
}

// NewLinear создает линейный слой с весами (outDim×inDim) и смещениями outDim.
func NewLinear(inDim, outDim int) *Linear {
	return &Linear{
		InDim:  inDim,
		OutDim: outDim,
		W:      randMat(outDim, inDim, 0.1),
		B:      randVec(outDim, 0.1),
	}
}

// Forward сохраняет вход для backward, а возвращает out = W*x + B
func (l *Linear) Forward(x []float64) []float64 {
	l.X = copyVec(x)
	return addVec(matVecMul(l.W, x), l.B)
}

// Backward принимает dout (dL/dOut), возвращает dX (dL/dX), а также dW и dB.
func (l *Linear) Backward(dout []float64) (dx []float64, dW [][]float64, dB []float64) {
	dx = make([]float64, l.InDim)
	dW = make([][]float64, l.OutDim)
	for i := 0; i < l.OutDim; i++ {
		dW[i] = make([]float64, l.InDim)
	}
	dB = make([]float64, l.OutDim)
	// dx = W^T * dout
	for i := 0; i < l.InDim; i++ {
		s := 0.0
		for j := 0; j < l.OutDim; j++ {
			s += l.W[j][i] * dout[j]
		}
		dx[i] = s
	}
	// dW = outer(dout, X)
	for j := 0; j < l.OutDim; j++ {
		for i := 0; i < l.InDim; i++ {
			dW[j][i] = dout[j] * l.X[i]
		}
	}
	// dB = dout (просто копия)
	copy(dB, dout)
	return
}

// 4.2. EmbeddingLayer
type EmbeddingLayer struct {
	VocabSize int
	EmbedSize int
	TokenEmbed [][]float64 // VocabSize x EmbedSize
	PosEmbed   [][]float64 // maxSeq x EmbedSize
	Tokens     []int
}

// NewEmbeddingLayer создает слой токенных и позиционных эмбеддингов.
func NewEmbeddingLayer(vocabSize, embedSize, maxSeq int) *EmbeddingLayer {
	return &EmbeddingLayer{
		VocabSize:  vocabSize,
		EmbedSize:  embedSize,
		TokenEmbed: randMat(vocabSize, embedSize, 0.1),
		PosEmbed:   randMat(maxSeq, embedSize, 0.1),
	}
}

// Forward берет входные индексы tokens, возвращает slice размером T x EmbedSize
// (T = len(tokens)).
func (emb *EmbeddingLayer) Forward(tokens []int) [][]float64 {
	emb.Tokens = make([]int, len(tokens))
	copy(emb.Tokens, tokens)
	out := make([][]float64, len(tokens))
	for t, tok := range tokens {
		vec := make([]float64, emb.EmbedSize)
		for i := 0; i < emb.EmbedSize; i++ {
			vec[i] = emb.TokenEmbed[tok][i] + emb.PosEmbed[t][i]
		}
		out[t] = vec
	}
	return out
}

// Backward — упрощённая функция, которая возвращала бы dToken, dPos. Не реализуем здесь полноценно.
func (emb *EmbeddingLayer) Backward(dout [][]float64) {
	// Здесь можно накапливать градиенты по TokenEmbed и PosEmbed.
}

// 4.3. LayerNorm (упрощенная версия с Forward и Backward)
type LayerNorm struct {
	Dim   int
	Gamma []float64
	Beta  []float64
	Eps   float64
	X     []float64
	Mean  float64
	Var   float64
	NormX []float64
}

func NewLayerNorm(dim int) *LayerNorm {
	g := make([]float64, dim)
	b := make([]float64, dim)
	for i := 0; i < dim; i++ {
		g[i] = 1.0
		b[i] = 0.0
	}
	return &LayerNorm{
		Dim:   dim,
		Gamma: g,
		Beta:  b,
		Eps:   1e-5,
	}
}

func (ln *LayerNorm) Forward(x []float64) []float64 {
	ln.X = copyVec(x)
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(ln.Dim)
	ln.Mean = mean
	variance := 0.0
	for _, v := range x {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(ln.Dim)
	ln.Var = variance
	out := make([]float64, ln.Dim)
	ln.NormX = make([]float64, ln.Dim)
	for i, v := range x {
		ln.NormX[i] = (v - mean) / math.Sqrt(variance+ln.Eps)
		out[i] = ln.Gamma[i]*ln.NormX[i] + ln.Beta[i]
	}
	return out
}

func (ln *LayerNorm) Backward(dout []float64) ([]float64, []float64, []float64) {
	N := float64(ln.Dim)
	dGamma := make([]float64, ln.Dim)
	dBeta := make([]float64, ln.Dim)
	for i := 0; i < ln.Dim; i++ {
		dGamma[i] = dout[i] * ln.NormX[i]
		dBeta[i] = dout[i]
	}
	dnorm := make([]float64, ln.Dim)
	for i := 0; i < ln.Dim; i++ {
		dnorm[i] = dout[i] * ln.Gamma[i]
	}
	invStd := 1.0 / math.Sqrt(ln.Var+ln.Eps)
	sumDnorm := 0.0
	sumDnormNorm := 0.0
	for i := 0; i < ln.Dim; i++ {
		sumDnorm += dnorm[i]
		sumDnormNorm += dnorm[i] * ln.NormX[i]
	}
	dx := make([]float64, ln.Dim)
	for i := 0; i < ln.Dim; i++ {
		dx[i] = invStd / N * (N*dnorm[i] - sumDnorm - ln.NormX[i]*sumDnormNorm)
	}
	return dx, dGamma, dBeta
}

// 4.4. SelfAttention (одноголовое) — упрощённый вариант
type SelfAttention struct {
	EmbedSize int
}

func NewSelfAttention(embedSize int) *SelfAttention {
	return &SelfAttention{
		EmbedSize: embedSize,
	}
}

// Forward (упрощённо)
func (sa *SelfAttention) Forward(x [][]float64) ([][]float64, interface{}) {
	out := make([][]float64, len(x))
	for i := range x {
		out[i] = copyVec(x[i]) // Упрощённо не считаем Q,K,V.
	}
	return out, nil
}

func (sa *SelfAttention) Backward(dout [][]float64, cache interface{}) (dx [][]float64) {
	// Упрощённая реализация.
	dx = make([][]float64, len(dout))
	for i := range dout {
		dx[i] = copyVec(dout[i])
	}
	return
}

// 4.5. MLPBlock (двухслойный перцептрон) — упрощенный
type MLPBlock struct {
	Linear1 *Linear
	CacheZ1 []float64
	Linear2 *Linear
}

func NewMLPBlock(inDim, mlpDim int) *MLPBlock {
	return &MLPBlock{
		Linear1: NewLinear(inDim, mlpDim),
		Linear2: NewLinear(mlpDim, inDim),
	}
}

// ====================================================================================
// 5. Cross-Entropy Loss
// ====================================================================================

func crossEntropy(logits []float64, target int) float64 {
	probs := softmaxForward(logits)
	return -math.Log(probs[target] + 1e-9)
}

func crossEntropyGrad(logits []float64, target int) []float64 {
	probs := softmaxForward(logits)
	grad := make([]float64, len(probs))
	for i := range probs {
		grad[i] = probs[i]
	}
	grad[target] -= 1.0
	return grad
}

// ====================================================================================
// 6. Основная структура TransformerModel (один блок)
// ====================================================================================

type TransformerModel struct {
	Cfg   TransformerConfig
	Embed *EmbeddingLayer
	// Один блок (SelfAttention + MLP + LayerNorm).
	Block *TransformerBlock
	// Финальный линейный слой
	Final *Linear
}

// TransformerBlock объединяет SelfAttention, LayerNorm, MLP и ещё один LayerNorm.
type TransformerBlock struct {
	Attn   *SelfAttention
	AttnLN *LayerNorm
	MLP    *MLPBlock
	MLPLN  *LayerNorm
}

func NewTransformerBlock(embedSize, mlpDim int) *TransformerBlock {
	return &TransformerBlock{
		Attn:   NewSelfAttention(embedSize),
		AttnLN: NewLayerNorm(embedSize),
		MLP:    NewMLPBlock(embedSize, mlpDim),
		MLPLN:  NewLayerNorm(embedSize),
	}
}

func NewTransformerModel(cfg TransformerConfig) *TransformerModel {
	return &TransformerModel{
		Cfg:   cfg,
		Embed: NewEmbeddingLayer(cfg.VocabSize, cfg.EmbedSize, cfg.BlockSize),
		Block: NewTransformerBlock(cfg.EmbedSize, cfg.MLPDim),
		Final: NewLinear(cfg.EmbedSize, cfg.VocabSize),
	}
}

// Forward выполняет forward-проход: Embedding → TransformerBlock → Final Linear → logits
func (m *TransformerModel) Forward(tokens []int) ([][]float64, map[string]interface{}) {
	cache := make(map[string]interface{})
	// 1. Embedding
	embOut := m.Embed.Forward(tokens)
	cache["embOut"] = embOut
	// 2. TransformerBlock (упрощенно)
	//    Можно реализовать Residual + LN + SelfAttention + Residual + LN + MLP
	blockOut := embOut // упрощенно
	cache["blockOut"] = blockOut
	// 3. Финальный слой
	T := len(blockOut)
	logits := make([][]float64, T)
	for t := 0; t < T; t++ {
		logits[t] = m.Final.Forward(blockOut[t])
	}
	cache["logits"] = logits
	return logits, cache
}

// Backward — упрощённая функция, вызывается на dlogits и обновляет параметры финального слоя.
func (m *TransformerModel) Backward(dlogits [][]float64, cache map[string]interface{}) {
	blockOut := cache["blockOut"].([][]float64)
	T := len(blockOut)
	for t := 0; t < T; t++ {
		// Если dlogits nil, создаём градиент для cross-entropy (target=0).
		dout := crossEntropyGrad(cache["logits"].([][]float64)[t], 0)
		dx, dW, dB := m.Final.Backward(dout)
		// Обновляем финальный слой
		for j := 0; j < m.Final.OutDim; j++ {
			for i := 0; i < m.Final.InDim; i++ {
				m.Final.W[j][i] -= m.Cfg.LearningRate * dW[j][i]
			}
			m.Final.B[j] -= m.Cfg.LearningRate * dB[j]
		}
		_ = dx
	}
}

// ====================================================================================
// 7. Многопоточность: TrainTask, GradResult, trainWorker, trainParallel, saver
// ====================================================================================

type TrainTask struct{}
type GradResult struct {
	Loss float64
}

// trainWorker получает задачи из tasks канала, обрабатывает их (forward/backward),
// и отправляет GradResult (среднюю потерю) в results.
func trainWorker(model *TransformerModel, data []int, cfg TransformerConfig, results chan<- GradResult, tasks chan TrainTask, stop chan struct{}) {
	N := len(data)
	for {
		select {
		case _, ok := <-stop:
			// Если канал остановки закрыт, выходим.
			if !ok {
				return
			} else {
			 	return
			}
		case _, ok := <-tasks:
			// Если канал задач закрыт, выходим.
			if !ok {
				return
			}
			// Случайно выбираем фрагмент текста
			start := rand.Intn(N - cfg.BlockSize - 1)
			inputSeq := data[start : start+cfg.BlockSize]
			targetSeq := data[start+1 : start+cfg.BlockSize+1]
			// Forward
			logits, cache := model.Forward(inputSeq)
			// Считаем cross-entropy loss
			loss := 0.0
			for t := 0; t < len(logits); t++ {
				loss += crossEntropy(logits[t], targetSeq[t])
			}
			avgLoss := loss / float64(len(logits))
			// Backward
			model.Backward(nil, cache)
			// Отправляем результат
			results <- GradResult{Loss: avgLoss}
		}
	}
}

// saver слушает канал saveChan и сохраняет модель при получении сигнала (struct{}).
func saver(saveChan chan struct{}, model *TransformerModel, filename string) {
	for range saveChan {
		saveModel(model, filename)
		fmt.Println("Модель сохранена (checkpoint).")
	}
}

// trainParallel организует многопоточную тренировку.
// - Создаёт пул воркеров
// - В главном цикле отправляет задачи (TrainTask) и получает результаты (GradResult)
// - По сигналу Ctrl+C прерывает обучение и сохраняет модель
func trainParallel(model *TransformerModel, data []int, cfg TransformerConfig, idx2char map[int]rune) {
	numIters := cfg.NumIterations
	checkpointInterval := 1000

	tasksChan := make(chan TrainTask)
	resultsChan := make(chan GradResult)
	stopChan := make(chan struct{})
	saveChan := make(chan struct{})

	// Запускаем горутину-сейвер
	go saver(saveChan, model, "model_checkpoint.json")

	numWorkers := runtime.NumCPU()
	fmt.Printf("Запущено %d воркеров.\n", numWorkers)
	// Создаём pool воркеров
	for i := 0; i < numWorkers; i++ {
		go trainWorker(model, data, cfg, resultsChan, tasksChan, stopChan)
	}

	// Канал для сигналов (SIGINT/SIGTERM)
	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, syscall.SIGTERM)

	totalLoss := 0.0
	for iter := 1; iter <= numIters; iter++ {
		select {
		case <-sigChan:
			fmt.Println("Сигнал завершения. Сохраняю модель и завершаю обучение.")
			close(stopChan)
			fmt.Println("1")
			break
			fmt.Println("2")
		default:
		}
		// Отправляем задачу
		tasksChan <- TrainTask{}
		// Получаем результат
		res := <-resultsChan
		totalLoss += res.Loss
		// Каждые 100 итераций выводим среднюю потерю и пример генерации
		if iter%100 == 0 {
			avgLoss := totalLoss / 100.0
			fmt.Printf("Итерация %d, Средняя потеря: %.4f\n", iter, avgLoss)
			totalLoss = 0.0
			// Пример генерации
			sample := generateText(model, data[:10], 200, idx2char, 0.8, 5)
			fmt.Println("Пример генерации:")
			fmt.Println(sample)
			fmt.Println("--------------------------")
		}
		// Каждые checkpointInterval итераций вызываем сейвер
		if iter%checkpointInterval == 0 {
			saveChan <- struct{}{}
		}
	}
	// Закрываем канал задач, сохраним модель
	//close(tasksChan)
	fmt.Println("Сохраняем модель в model.json")
	saveModel(model, "model.json")
	fmt.Println("Обучение завершено. Модель сохранена в model.json")
}

// ====================================================================================
// 8. Генерация текста (побуквенно) и режим вопрос-ответ (ask)
// ====================================================================================

// generateText генерирует text посимвольно, выводя символ сразу после генерации.
func generateText(model *TransformerModel, startTokens []int, length int, idx2char map[int]rune, temperature float64, topK int) string {
	if len(startTokens) == 0 {
		startTokens = []int{0}
	}
	tokens := make([]int, len(startTokens))
	copy(tokens, startTokens)
	fmt.Println("Начало генерации:")
	for i := 0; i < length; i++ {
		start := 0
		if len(tokens) > model.Cfg.BlockSize {
			start = len(tokens) - model.Cfg.BlockSize
		}
		input := tokens[start:]
		logits, _ := model.Forward(input)
		lastLogits := logits[len(logits)-1]
		probs := softmaxForward(lastLogits)
		nextToken := sampleFromDistribution(probs, temperature, topK)
		tokens = append(tokens, nextToken)
		ch := idx2char[nextToken]
		fmt.Print(string(ch))
		time.Sleep(20 * time.Millisecond)
		if ch == '.' || ch == '!' || ch == '?' {
			break
		}
	}
	fmt.Println()
	var sb strings.Builder
	for _, tok := range tokens {
		ch, ok := idx2char[tok]
		if !ok {
			ch = ' '
		}
		sb.WriteRune(ch)
	}
	return sb.String()
}

// sampleFromDistribution выбирает элемент по распределению probs с учётом topK и temperature.
func sampleFromDistribution(probs []float64, temperature float64, topK int) int {
	if topK < 1 {
		topK = len(probs)
	}
	// Повышаем/понижаем все вероятности, затем нормируем
	adjusted := make([]float64, len(probs))
	for i, p := range probs {
		adjusted[i] = math.Pow(p, 1.0/temperature)
	}
	adjusted = softmaxForward(adjusted)
	type Pair struct {
		Index int
		Prob  float64
	}
	var pairs []Pair
	for i, p := range adjusted {
		pairs = append(pairs, Pair{i, p})
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Prob > pairs[j].Prob
	})
	if topK > len(pairs) {
		topK = len(pairs)
	}
	total := 0.0
	for i := 0; i < topK; i++ {
		total += pairs[i].Prob
	}
	rnd := rand.Float64() * total
	sum := 0.0
	for i := 0; i < topK; i++ {
		sum += pairs[i].Prob
		if rnd < sum {
			return pairs[i].Index
		}
	}
	return pairs[topK-1].Index
}

// askQuestion принимает вопрос (строку), трансформирует её в токены, и генерирует ответ.
func askQuestion(model *TransformerModel, question string, char2idx map[rune]int, idx2char map[int]rune) string {
	tokens := []int{}
	for _, ch := range question {
		if idx, ok := char2idx[ch]; ok {
			tokens = append(tokens, idx)
		} else {
			// Если символ не найден в словаре, ставим 0 (обычно это padding или unknown).
			tokens = append(tokens, 0)
		}
	}
	answer := generateText(model, tokens, 200, idx2char, 0.8, 5)
	return answer
}

// ====================================================================================
// 9. Загрузка датасета, сохранение/загрузка модели, main
// ====================================================================================

// loadDataset читает tinyshakespeare.txt, строит словарь символов, превращает текст в индексы.
func loadDataset(path string) ([]int, map[rune]int, map[int]rune) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("Ошибка чтения файла: %v", err)
	}
	text := string(content)
	charSet := make(map[rune]bool)
	for _, ch := range text {
		charSet[ch] = true
	}
	char2idx := make(map[rune]int)
	idx2char := make(map[int]rune)
	idx := 0
	for ch := range charSet {
		char2idx[ch] = idx
		idx2char[idx] = ch
		idx++
	}
	data := []int{}
	for _, ch := range text {
		data = append(data, char2idx[ch])
	}
	return data, char2idx, idx2char
}

// saveModel сериализует TransformerModel в JSON.
func saveModel(model *TransformerModel, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatalf("Ошибка создания файла: %v", err)
	}
	defer file.Close()
	enc := json.NewEncoder(file)
	if err := enc.Encode(model); err != nil {
		log.Fatalf("Ошибка сохранения модели: %v", err)
	}
}

// loadModel десериализует TransformerModel из JSON.
func loadModel(filename string) (*TransformerModel, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var model TransformerModel
	dec := json.NewDecoder(file)
	if err := dec.Decode(&model); err != nil {
		return nil, err
	}
	return &model, nil
}

// main — точка входа. Определяет режим работы (train или ask), загружает модель и запускает процесс.
func main() {
	rand.Seed(time.Now().UnixNano())
	runtime.GOMAXPROCS(runtime.NumCPU())

	mode := flag.String("mode", "train", "Режим работы: 'train' или 'ask'")
	flag.Parse()

	datasetPath := "tinyshakespeare.txt"
	if _, err := os.Stat(datasetPath); err != nil {
		log.Fatalf("Файл %s не найден.", datasetPath)
	}
	data, char2idx, idx2char := loadDataset(datasetPath)
	fmt.Printf("Датасет загружен: %d символов, размер словаря: %d\n", len(data), len(char2idx))

	cfg := TransformerConfig{
		VocabSize:     len(char2idx),
		EmbedSize:     128,
		BlockSize:     128,
		MLPDim:        256,
		LearningRate:  0.0005,
		NumIterations: 3000,
		BatchSize:     64,
	}

	var model *TransformerModel
	m, err := loadModel("model.json")
	if err != nil {
		fmt.Println("Модель не найдена, создаём новую модель...")
		model = NewTransformerModel(cfg)
	} else {
		fmt.Println("Модель загружена!")
		model = m
		model.Cfg = cfg
	}

	if *mode == "ask" {
		reader := bufio.NewReader(os.Stdin)
		fmt.Println("Введите вопрос на английском:")
		q, _ := reader.ReadString('\n')
		q = strings.TrimSpace(q)
		answer := askQuestion(model, q, char2idx, idx2char)
		fmt.Println("Ответ в стиле Шекспира:")
		fmt.Println(answer)
	} else {
		fmt.Println("Начинается обучение модели...")
		trainParallel(model, data, cfg, idx2char)
		fmt.Println("Генерация финального текста:")
		finalText := generateText(model, []int{0}, 200, idx2char, 0.8, 5)
		fmt.Println(finalText)
	}
}

