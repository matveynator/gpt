// gpt.go
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
	"sort"
	"strings"
	"time"
)

// =============================================================================
// 1. Основные матричные операции и функции активации
// =============================================================================

func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		log.Fatalf("dotProduct: размеры векторов не совпадают: %d vs %d", len(a), len(b))
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func matMul(A, B [][]float64) [][]float64 {
	m := len(A)
	n := len(A[0])
	p := len(B[0])
	C := make([][]float64, m)
	for i := 0; i < m; i++ {
		C[i] = make([]float64, p)
		for j := 0; j < p; j++ {
			s := 0.0
			for k := 0; k < n; k++ {
				s += A[i][k] * B[k][j]
			}
			C[i][j] = s
		}
	}
	return C
}

func matVecMul(A [][]float64, x []float64) []float64 {
	m := len(A)
	n := len(A[0])
	if len(x) != n {
		log.Fatalf("matVecMul: размер матрицы %dx%d и длина вектора %d", m, n, len(x))
	}
	out := make([]float64, m)
	for i := 0; i < m; i++ {
		s := 0.0
		for j := 0; j < n; j++ {
			s += A[i][j] * x[j]
		}
		out[i] = s
	}
	return out
}

func addVec(a, b []float64) []float64 {
	if len(a) != len(b) {
		log.Fatalf("addVec: длины векторов не совпадают: %d vs %d", len(a), len(b))
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func relu(x []float64) []float64 {
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

func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x {
		if v > maxVal {
			maxVal = v
		}
	}
	expSum := 0.0
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = math.Exp(v - maxVal)
		expSum += out[i]
	}
	if expSum == 0 {
		expSum = 1e-9
	}
	for i := range out {
		out[i] /= expSum
	}
	return out
}

// =============================================================================
// 2. Компоненты модели: Embedding, LayerNorm, Self-Attention, MLP
// =============================================================================

// 2.1. Embedding: объединяет токенные и позиционные эмбеддинги.
type Embedding struct {
	TokenEmbedding      [][]float64 // VocabSize x EmbedSize
	PositionalEmbedding [][]float64 // MaxSeqLen x EmbedSize
}

func NewEmbedding(vocabSize, embedSize, maxSeqLen int) *Embedding {
	return &Embedding{
		TokenEmbedding:      newRandomMatrix(vocabSize, embedSize),
		PositionalEmbedding: newRandomMatrix(maxSeqLen, embedSize),
	}
}

func (e *Embedding) Apply(tokens []int) [][]float64 {
	seqLen := len(tokens)
	out := make([][]float64, seqLen)
	for t, token := range tokens {
		vec := make([]float64, len(e.TokenEmbedding[0]))
		for i := 0; i < len(vec); i++ {
			vec[i] = e.TokenEmbedding[token][i] + e.PositionalEmbedding[t][i]
		}
		out[t] = vec
	}
	return out
}

// 2.2. LayerNorm: нормализует входной вектор.
type LayerNorm struct {
	Gamma []float64 // Масштабирование
	Beta  []float64 // Смещение
	Eps   float64   // Для числовой стабильности
}

func NewLayerNorm(dim int) *LayerNorm {
	gamma := make([]float64, dim)
	beta := make([]float64, dim)
	for i := 0; i < dim; i++ {
		gamma[i] = 1.0
		beta[i] = 0.0
	}
	return &LayerNorm{Gamma: gamma, Beta: beta, Eps: 1e-5}
}

func (ln *LayerNorm) Apply(x []float64) []float64 {
	mean := 0.0
	for _, v := range x {
		mean += v
	}
	mean /= float64(len(x))
	variance := 0.0
	for _, v := range x {
		variance += (v - mean) * (v - mean)
	}
	variance /= float64(len(x))
	out := make([]float64, len(x))
	for i, v := range x {
		norm := (v - mean) / math.Sqrt(variance+ln.Eps)
		out[i] = ln.Gamma[i]*norm + ln.Beta[i]
	}
	return out
}

// 2.3. Self-Attention (одноголовое)
type SelfAttention struct {
	Wq [][]float64 // EmbedSize x EmbedSize
	Wk [][]float64
	Wv [][]float64
	Wo [][]float64
}

func NewSelfAttention(embedSize int) *SelfAttention {
	return &SelfAttention{
		Wq: newRandomMatrix(embedSize, embedSize),
		Wk: newRandomMatrix(embedSize, embedSize),
		Wv: newRandomMatrix(embedSize, embedSize),
		Wo: newRandomMatrix(embedSize, embedSize),
	}
}

func (sa *SelfAttention) Apply(x [][]float64) [][]float64 {
	seqLen := len(x)
	embedSize := len(x[0])
	Q := make([][]float64, seqLen)
	K := make([][]float64, seqLen)
	V := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		Q[t] = matVecMul(sa.Wq, x[t])
		K[t] = matVecMul(sa.Wk, x[t])
		V[t] = matVecMul(sa.Wv, x[t])
	}
	out := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		scores := make([]float64, seqLen)
		for i := 0; i < seqLen; i++ {
			scores[i] = dotProduct(Q[t], K[i])
		}
		scale := math.Sqrt(float64(embedSize))
		for i := range scores {
			scores[i] /= scale
		}
		attnWeights := softmax(scores)
		attnOut := make([]float64, embedSize)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < embedSize; j++ {
				attnOut[j] += attnWeights[i] * V[i][j]
			}
		}
		out[t] = matVecMul(sa.Wo, attnOut)
	}
	return out
}

// 2.4. MLP-блок: двухслойный перцептрон с ReLU.
type MLP struct {
	W1 [][]float64 // MLPDim x EmbedSize
	B1 []float64   // MLPDim
	W2 [][]float64 // EmbedSize x MLPDim
	B2 []float64   // EmbedSize
}

func NewMLP(embedSize, mlpDim int) *MLP {
	return &MLP{
		W1: newRandomMatrix(mlpDim, embedSize),
		B1: newRandomVector(mlpDim),
		W2: newRandomMatrix(embedSize, mlpDim),
		B2: newRandomVector(embedSize),
	}
}

func (mlp *MLP) Apply(x [][]float64) [][]float64 {
	seqLen := len(x)
	out := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		z1 := addVec(matVecMul(mlp.W1, x[t]), mlp.B1)
		a1 := relu(z1)
		out[t] = addVec(matVecMul(mlp.W2, a1), mlp.B2)
	}
	return out
}

// =============================================================================
// 3. Transformer Block: объединяет самовнимание, остаточные связи и MLP с LayerNorm
// =============================================================================

type TransformerBlock struct {
	Attention *SelfAttention
	AttnNorm  *LayerNorm
	MLP       *MLP
	MLPNorm   *LayerNorm
}

func NewTransformerBlock(embedSize, mlpDim int) *TransformerBlock {
	return &TransformerBlock{
		Attention: NewSelfAttention(embedSize),
		AttnNorm:  NewLayerNorm(embedSize),
		MLP:       NewMLP(embedSize, mlpDim),
		MLPNorm:   NewLayerNorm(embedSize),
	}
}

func (tb *TransformerBlock) Apply(x [][]float64) [][]float64 {
	attnOut := tb.Attention.Apply(x)
	res1 := make([][]float64, len(x))
	for i := range x {
		res1[i] = tb.AttnNorm.Apply(addVec(x[i], attnOut[i]))
	}
	mlpOut := tb.MLP.Apply(res1)
	res2 := make([][]float64, len(res1))
	for i := range res1 {
		res2[i] = tb.MLPNorm.Apply(addVec(res1[i], mlpOut[i]))
	}
	return res2
}

// =============================================================================
// 4. Transformer Model: объединение эмбеддингов, блоков и финального слоя
// =============================================================================

type TransformerConfig struct {
	VocabSize     int     // Размер словаря
	EmbedSize     int     // Размер эмбеддингов
	BlockSize     int     // Максимальная длина входной последовательности
	NumHeads      int     // Количество голов (используем 1 для простоты)
	MLPDim        int     // Размер скрытого слоя в MLP
	LearningRate  float64 // Скорость обучения
	NumIterations int     // Количество итераций обучения
	BatchSize     int     // Размер батча (не используется)
}

type TransformerModel struct {
	Cfg         TransformerConfig   `json:"cfg"`
	Embed       *Embedding          `json:"embed"`
	Blocks      []*TransformerBlock `json:"blocks"`
	FinalLinear [][]float64         `json:"finalLinear"` // VocabSize x EmbedSize
	FinalBias   []float64           `json:"finalBias"`   // VocabSize
}

func NewTransformerModel(cfg TransformerConfig, numBlocks int) *TransformerModel {
	embed := NewEmbedding(cfg.VocabSize, cfg.EmbedSize, cfg.BlockSize)
	blocks := make([]*TransformerBlock, numBlocks)
	for i := 0; i < numBlocks; i++ {
		blocks[i] = NewTransformerBlock(cfg.EmbedSize, cfg.MLPDim)
	}
	finalLinear := newRandomMatrix(cfg.VocabSize, cfg.EmbedSize)
	finalBias := newRandomVector(cfg.VocabSize)
	return &TransformerModel{
		Cfg:         cfg,
		Embed:       embed,
		Blocks:      blocks,
		FinalLinear: finalLinear,
		FinalBias:   finalBias,
	}
}

func (m *TransformerModel) Forward(tokens []int) [][]float64 {
	x := m.Embed.Apply(tokens)
	for _, block := range m.Blocks {
		x = block.Apply(x)
	}
	seqLen := len(x)
	logits := make([][]float64, seqLen)
	for t := 0; t < seqLen; t++ {
		logits[t] = addVec(matVecMul(m.FinalLinear, x[t]), m.FinalBias)
	}
	return logits
}

// =============================================================================
// 5. Вспомогательные функции для случайных матриц и векторов
// =============================================================================

func newRandomMatrix(rows, cols int) [][]float64 {
	mat := make([][]float64, rows)
	for i := 0; i < rows; i++ {
		mat[i] = make([]float64, cols)
		for j := 0; j < cols; j++ {
			mat[i][j] = rand.Float64()*0.2 - 0.1
		}
	}
	return mat
}

func newRandomVector(n int) []float64 {
	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		vec[i] = rand.Float64()*0.2 - 0.1
	}
	return vec
}

// =============================================================================
// 6. Функция потерь (Cross-Entropy Loss)
// =============================================================================

func crossEntropyLoss(logits []float64, target int) float64 {
	probs := softmax(logits)
	loss := -math.Log(probs[target] + 1e-9)
	return loss
}

// =============================================================================
// 7. Генерация текста (Inference)
// =============================================================================

func generateText(model *TransformerModel, startTokens []int, length int, idx2char map[int]rune, temperature float64, topK int) string {
	if len(startTokens) == 0 {
		startTokens = []int{0}
	}
	tokens := make([]int, len(startTokens))
	copy(tokens, startTokens)
	for i := 0; i < length; i++ {
		start := 0
		if len(tokens) > model.Cfg.BlockSize {
			start = len(tokens) - model.Cfg.BlockSize
		}
		input := tokens[start:]
		logits := model.Forward(input)
		lastLogits := logits[len(logits)-1]
		probs := softmax(lastLogits)
		nextToken := sampleFromDistribution(probs, temperature, topK)
		tokens = append(tokens, nextToken)
		ch := idx2char[nextToken]
		if ch == '.' || ch == '!' || ch == '?' {
			break
		}
	}
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

func sampleFromDistribution(probs []float64, temperature float64, topK int) int {
	if topK < 1 {
		topK = len(probs)
	}
	adjusted := make([]float64, len(probs))
	for i, p := range probs {
		adjusted[i] = math.Pow(p, 1.0/temperature)
	}
	adjusted = softmax(adjusted)
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

// =============================================================================
// 8. Загрузка датасета (TinyShakespeare)
// =============================================================================

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
	data := make([]int, 0, len(text))
	for _, ch := range text {
		data = append(data, char2idx[ch])
	}
	return data, char2idx, idx2char
}

// =============================================================================
// 9. Сохранение и загрузка модели (JSON)
// =============================================================================

func saveModel(model *TransformerModel, filename string) {
	file, err := os.Create(filename)
	if err != nil {
		log.Fatalf("Ошибка при создании файла: %v", err)
	}
	defer file.Close()
	encoder := json.NewEncoder(file)
	if err := encoder.Encode(model); err != nil {
		log.Fatalf("Ошибка при сохранении модели: %v", err)
	}
}

func loadModel(filename string) (*TransformerModel, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()
	var model TransformerModel
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&model); err != nil {
		return nil, err
	}
	return &model, nil
}

// =============================================================================
// 10. Обучение модели (Skeleton)
// =============================================================================

// Здесь реализован skeleton обучения (forward‑проход и вычисление потерь).
// Обратное распространение не реализовано полностью.
// Кроме того, модель сохраняется периодически (каждые 1000 итераций).
func train(model *TransformerModel, data []int, char2idx map[rune]int, idx2char map[int]rune) {
	cfg := model.Cfg
	N := len(data)
	numIters := cfg.NumIterations

	for iter := 0; iter < numIters; iter++ {
		start := rand.Intn(N - cfg.BlockSize - 1)
		inputSeq := data[start : start+cfg.BlockSize]
		targetSeq := data[start+1 : start+cfg.BlockSize+1]

		logits := model.Forward(inputSeq)
		totalLoss := 0.0
		for t := 0; t < len(logits); t++ {
			totalLoss += crossEntropyLoss(logits[t], targetSeq[t])
		}
		avgLoss := totalLoss / float64(len(logits))

		if iter%100 == 0 {
			fmt.Printf("Итерация %d, Средняя потеря: %.4f\n", iter, avgLoss)
			sample := generateText(model, inputSeq[:10], 200, idx2char, 0.8, 5)
			fmt.Println("Пример сгенерированного текста:")
			fmt.Println(sample)
			fmt.Println("--------------------------")
		}

		// Здесь можно добавить backpropagation и обновление параметров.

		// Сохраняем модель каждые 1000 итераций.
		if iter > 0 && iter%1000 == 0 {
			saveModel(model, "model.json")
			fmt.Printf("Модель сохранена на итерации %d\n", iter)
		}
	}

	saveModel(model, "model.json")
}

// =============================================================================
// 11. Функция для ответа на вопрос (инференс)
// =============================================================================

// askQuestion принимает вопрос на английском, преобразует его в токены и генерирует ответ в стиле Шекспира.
func askQuestion(model *TransformerModel, question string, char2idx map[rune]int, idx2char map[int]rune) string {
	// Преобразуем вопрос в последовательность токенов.
	// Для каждого символа в вопросе, если его нет в словаре, используем индекс 0.
	tokens := []int{}
	for _, ch := range question {
		if idx, ok := char2idx[ch]; ok {
			tokens = append(tokens, idx)
		} else {
			tokens = append(tokens, 0)
		}
	}
	// Генерируем ответ с помощью модели.
	answer := generateText(model, tokens, 200, idx2char, 0.8, 5)
	return answer
}

// =============================================================================
// 12. Главная функция main
// =============================================================================

func main() {
	rand.Seed(time.Now().UnixNano())

	// Флаг для режима вопрос-ответ.
	mode := flag.String("mode", "train", "Режим: 'train' для обучения, 'ask' для вопроса")
	flag.Parse()

	datasetPath := "tinyshakespeare.txt"
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		log.Fatalf("Файл %s не найден. Поместите датасет в рабочую директорию.", datasetPath)
	}

	data, char2idx, idx2char := loadDataset(datasetPath)
	fmt.Printf("Датасет загружен: %d символов, размер словаря: %d\n", len(data), len(char2idx))

	cfg := TransformerConfig{
		VocabSize:     len(char2idx),
		EmbedSize:     128,
		BlockSize:     128,
		NumHeads:      1,
		MLPDim:        256,
		LearningRate:  0.0005,
		NumIterations: 15000,
		BatchSize:     64,
	}

	numBlocks := 6
	var model *TransformerModel
	if *mode == "ask" {
		// Если режим "ask", пытаемся загрузить модель.
		m, err := loadModel("model.json")
		if err != nil {
			log.Fatalf("Модель не найдена! Сначала обучите модель (mode=train)")
		}
		fmt.Println("Модель загружена!")
		model = m
		// Если позиционные эмбеддинги пустые, инициализируем заново.
		if len(model.Embed.PositionalEmbedding) == 0 {
			model.Embed.PositionalEmbedding = newRandomMatrix(cfg.BlockSize, cfg.EmbedSize)
		}
		model.Cfg = cfg
	} else {
		// Режим обучения.
		if _, err := os.Stat("model.json"); err != nil {
			fmt.Println("Сохранённая модель не найдена, создаём новую модель...")
			model = NewTransformerModel(cfg, numBlocks)
		} else {
			m, err := loadModel("model.json")
			if err != nil {
				fmt.Println("Ошибка загрузки модели, создаём новую модель...")
				model = NewTransformerModel(cfg, numBlocks)
			} else {
				fmt.Println("Модель загружена!")
				model = m
				if len(model.Embed.PositionalEmbedding) == 0 {
					model.Embed.PositionalEmbedding = newRandomMatrix(cfg.BlockSize, cfg.EmbedSize)
				}
			}
			model.Cfg = cfg
		}
	}

	if *mode == "ask" {
		// Режим вопрос-ответ.
		reader := bufio.NewReader(os.Stdin)
		fmt.Println("Введите ваш вопрос на английском:")
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)
		answer := askQuestion(model, question, char2idx, idx2char)
		fmt.Println("Ответ в стиле Шекспира:")
		fmt.Println(answer)
	} else {
		// Режим обучения.
		fmt.Println("Начинается обучение модели...")
		train(model, data, char2idx, idx2char)
		fmt.Println("Обучение завершено!")
		fmt.Println("Генерация финального текста:")
		finalText := generateText(model, []int{0}, 200, idx2char, 0.8, 5)
		fmt.Println(finalText)
	}
}

