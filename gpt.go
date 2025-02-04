// nanogpt2_improved.go
package main

import (
	"encoding/json"
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
// 1. Матричные операции и функции активации
// =============================================================================

// dotProduct вычисляет скалярное произведение двух векторов одинаковой длины.
func dotProduct(a, b []float64) float64 {
	if len(a) != len(b) {
		log.Fatalf("Размеры векторов не совпадают: %d vs %d", len(a), len(b))
	}
	sum := 0.0
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// matMul выполняет умножение матрицы A (m x n) на матрицу B (n x p) и возвращает матрицу (m x p).
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

// matVecMul умножает матрицу A (m x n) на вектор x (n) и возвращает вектор размера m.
func matVecMul(A [][]float64, x []float64) []float64 {
	m := len(A)
	n := len(A[0])
	if len(x) != n {
		log.Fatalf("Невозможно умножить: размер матрицы %dx%d и длина вектора %d", m, n, len(x))
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

// addVec складывает два вектора одинаковой длины.
func addVec(a, b []float64) []float64 {
	if len(a) != len(b) {
		log.Fatalf("Невозможно сложить векторы разной длины: %d и %d", len(a), len(b))
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// relu применяет ReLU поэлементно к вектору.
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

// reluBackward вычисляет градиент ReLU: если входной x > 0, градиент равен gradOut, иначе 0.
func reluBackward(x, gradOut []float64) []float64 {
	grad := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			grad[i] = gradOut[i]
		} else {
			grad[i] = 0
		}
	}
	return grad
}

// softmax применяет softmax-функцию к вектору.
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
// 2. Определение конфигурации и структуры модели Transformer
// =============================================================================

// TransformerConfig задаёт гиперпараметры модели.
type TransformerConfig struct {
	VocabSize     int     // Число уникальных символов (размер словаря)
	EmbedSize     int     // Размер эмбеддингов
	BlockSize     int     // Максимальная длина входной последовательности (контекст)
	NumHeads      int     // Число голов (здесь используется 1 для простоты)
	MLPDim        int     // Размер скрытого слоя в MLP-блоке
	LearningRate  float64 // Начальная скорость обучения
	NumIterations int     // Количество итераций обучения
	BatchSize     int     // Размер батча (не используется в данном примере)
}

// TransformerModel содержит все параметры модели.
// **Важно:** Все поля экспортируются (с заглавной буквы) для корректной сериализации через JSON.
type TransformerModel struct {
	Cfg                  TransformerConfig `json:"cfg"`
	Embeddings           [][]float64       `json:"embeddings"`           // Матрица эмбеддингов: VocabSize x EmbedSize
	PositionalEmbeddings [][]float64       `json:"positionalEmbeddings"` // Позиционные эмбеддинги: BlockSize x EmbedSize
	Wq                   [][]float64       `json:"Wq"`                   // Весовая матрица для запросов: EmbedSize x EmbedSize
	Wk                   [][]float64       `json:"Wk"`                   // Весовая матрица для ключей: EmbedSize x EmbedSize
	Wv                   [][]float64       `json:"Wv"`                   // Весовая матрица для значений: EmbedSize x EmbedSize
	Wo                   [][]float64       `json:"Wo"`                   // Весовая матрица для объединения: EmbedSize x EmbedSize
	W1                   [][]float64       `json:"W1"`                   // Весовая матрица первого слоя MLP: MLPDim x EmbedSize
	B1                   []float64         `json:"B1"`                   // Смещения первого слоя MLP: длина MLPDim
	W2                   [][]float64       `json:"W2"`                   // Весовая матрица второго слоя MLP: EmbedSize x MLPDim
	B2                   []float64         `json:"B2"`                   // Смещения второго слоя MLP: длина EmbedSize
	Wout                 [][]float64       `json:"Wout"`                 // Весовая матрица финального слоя: VocabSize x EmbedSize
	Bout                 []float64         `json:"Bout"`                 // Смещения финального слоя: длина VocabSize
}

// newRandomMatrix создаёт матрицу размером rows x cols с случайными значениями (от -0.1 до 0.1).
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

// newRandomVector создаёт вектор длины n с случайными значениями (от -0.1 до 0.1).
func newRandomVector(n int) []float64 {
	vec := make([]float64, n)
	for i := 0; i < n; i++ {
		vec[i] = rand.Float64()*0.2 - 0.1
	}
	return vec
}

// NewTransformerModel инициализирует новую модель с заданной конфигурацией.
func NewTransformerModel(cfg TransformerConfig) *TransformerModel {
	return &TransformerModel{
		Cfg:                  cfg,
		Embeddings:           newRandomMatrix(cfg.VocabSize, cfg.EmbedSize),
		PositionalEmbeddings: newRandomMatrix(cfg.BlockSize, cfg.EmbedSize),
		Wq:                   newRandomMatrix(cfg.EmbedSize, cfg.EmbedSize),
		Wk:                   newRandomMatrix(cfg.EmbedSize, cfg.EmbedSize),
		Wv:                   newRandomMatrix(cfg.EmbedSize, cfg.EmbedSize),
		Wo:                   newRandomMatrix(cfg.EmbedSize, cfg.EmbedSize),
		W1:                   newRandomMatrix(cfg.MLPDim, cfg.EmbedSize),
		B1:                   newRandomVector(cfg.MLPDim),
		W2:                   newRandomMatrix(cfg.EmbedSize, cfg.MLPDim),
		B2:                   newRandomVector(cfg.EmbedSize),
		Wout:                 newRandomMatrix(cfg.VocabSize, cfg.EmbedSize),
		Bout:                 newRandomVector(cfg.VocabSize),
	}
}

// =============================================================================
// 3. Прямой проход (Forward Propagation) с сохранением промежуточных значений (Cache)
// =============================================================================

// Cache хранит промежуточные результаты прямого прохода, необходимые для backpropagation.
type Cache struct {
	Embedded     [][]float64 `json:"embedded"`     // Эмбеддинги (с добавлением позиционных)
	AttentionOut [][]float64 `json:"attentionOut"` // Выход слоя самовнимания
	Z1           [][]float64 `json:"z1"`           // Линейное преобразование первого слоя MLP (до ReLU)
	A1           [][]float64 `json:"a1"`           // Выход ReLU первого слоя MLP
	MlpOut       [][]float64 `json:"mlpOut"`       // Выход MLP блока (вход для финального слоя)
	Logits       [][]float64 `json:"logits"`       // Логиты финального слоя (до softmax)
}

// forwardWithCache выполняет прямой проход по модели для входной последовательности токенов и сохраняет промежуточные результаты.
func (m *TransformerModel) forwardWithCache(tokens []int) ([][]float64, *Cache) {
	T := len(tokens)
	cache := &Cache{
		Embedded:     make([][]float64, T),
		AttentionOut: make([][]float64, T),
		Z1:           make([][]float64, T),
		A1:           make([][]float64, T),
		MlpOut:       make([][]float64, T),
		Logits:       make([][]float64, T),
	}

	// 1. Эмбеддинги + позиционные эмбеддинги.
	embedded := make([][]float64, T)
	for t, token := range tokens {
		// Если t выходит за пределы PositionalEmbeddings, завершаем с ошибкой.
		if t >= len(m.PositionalEmbeddings) {
			log.Fatalf("Ошибка: t (%d) >= длины PositionalEmbeddings (%d)", t, len(m.PositionalEmbeddings))
		}
		vec := make([]float64, m.Cfg.EmbedSize)
		for i := 0; i < m.Cfg.EmbedSize; i++ {
			vec[i] = m.Embeddings[token][i] + m.PositionalEmbeddings[t][i]
		}
		embedded[t] = vec
	}
	cache.Embedded = embedded

	// 2. Слой самовнимания (здесь не реализован backpropagation, просто forward).
	Q := make([][]float64, T)
	K := make([][]float64, T)
	V := make([][]float64, T)
	for t := 0; t < T; t++ {
		Q[t] = matVecMul(m.Wq, embedded[t])
		K[t] = matVecMul(m.Wk, embedded[t])
		V[t] = matVecMul(m.Wv, embedded[t])
	}
	attentionOut := make([][]float64, T)
	for t := 0; t < T; t++ {
		scores := make([]float64, T)
		for i := 0; i < T; i++ {
			scores[i] = dotProduct(Q[t], K[i])
		}
		scale := math.Sqrt(float64(m.Cfg.EmbedSize))
		for i := range scores {
			scores[i] /= scale
		}
		attnWeights := softmax(scores)
		outVec := make([]float64, m.Cfg.EmbedSize)
		for i := 0; i < T; i++ {
			for j := 0; j < m.Cfg.EmbedSize; j++ {
				outVec[j] += attnWeights[i] * V[i][j]
			}
		}
		attentionOut[t] = matVecMul(m.Wo, outVec)
	}
	cache.AttentionOut = attentionOut

	// 3. MLP блок.
	mlpOut := make([][]float64, T)
	z1 := make([][]float64, T)
	a1 := make([][]float64, T)
	for t := 0; t < T; t++ {
		// Первый линейный слой: z1 = W1 * attentionOut + B1.
		z := addVec(matVecMul(m.W1, attentionOut[t]), m.B1)
		z1[t] = z
		// Применяем ReLU.
		a := relu(z)
		a1[t] = a
		// Второй линейный слой: mlpOut = W2 * a + B2.
		mlpOut[t] = addVec(matVecMul(m.W2, a), m.B2)
	}
	cache.Z1 = z1
	cache.A1 = a1
	cache.MlpOut = mlpOut

	// 4. Финальный линейный слой для получения логитов.
	logits := make([][]float64, T)
	for t := 0; t < T; t++ {
		logits[t] = addVec(matVecMul(m.Wout, mlpOut[t]), m.Bout)
	}
	cache.Logits = logits

	return logits, cache
}

// =============================================================================
// 4. Backpropagation: вычисление градиентов и оптимизация (с использованием алгоритма Adam)
// =============================================================================

// Gradients хранит градиенты для обновления параметров (только для MLP и финального слоя).
type Gradients struct {
	dW1   [][]float64 // Размер: MLPDim x EmbedSize
	db1   []float64   // Длина: MLPDim
	dW2   [][]float64 // Размер: EmbedSize x MLPDim
	db2   []float64   // Длина: EmbedSize
	dWout [][]float64 // Размер: VocabSize x EmbedSize
	dbout []float64   // Длина: VocabSize
}

// zeroGradients инициализирует градиенты нулями.
func zeroGradients(cfg TransformerConfig) *Gradients {
	// Инициализируем матрицы нулями
	dW1 := make([][]float64, cfg.MLPDim)
	for i := 0; i < cfg.MLPDim; i++ {
		dW1[i] = make([]float64, cfg.EmbedSize)
	}
	dW2 := make([][]float64, cfg.EmbedSize)
	for i := 0; i < cfg.EmbedSize; i++ {
		dW2[i] = make([]float64, cfg.MLPDim)
	}
	dWout := make([][]float64, cfg.VocabSize)
	for i := 0; i < cfg.VocabSize; i++ {
		dWout[i] = make([]float64, cfg.EmbedSize)
	}
	return &Gradients{
		dW1:   dW1,
		db1:   make([]float64, cfg.MLPDim),
		dW2:   dW2,
		db2:   make([]float64, cfg.EmbedSize),
		dWout: dWout,
		dbout: make([]float64, cfg.VocabSize),
	}
}

// computeGradients вычисляет градиенты для одного примера (последовательности) по кросс-энтропии.
func computeGradients(model *TransformerModel, tokens []int, target []int, cache *Cache) *Gradients {
	grad := zeroGradients(model.Cfg)
	T := len(tokens)
	for t := 0; t < T; t++ {
		// 4.1. Вычисляем градиент логитов: dlogits = softmax(logits) - one-hot(target)
		probs := softmax(cache.Logits[t])
		onehot := make([]float64, model.Cfg.VocabSize)
		onehot[target[t]] = 1.0
		dlogits := make([]float64, model.Cfg.VocabSize)
		for i := 0; i < model.Cfg.VocabSize; i++ {
			dlogits[i] = probs[i] - onehot[i]
		}
		// 4.2. Градиенты для финального слоя:
		// dWout += outer(dlogits, MlpOut), dbout += dlogits
		for i := 0; i < model.Cfg.VocabSize; i++ {
			for j := 0; j < model.Cfg.EmbedSize; j++ {
				grad.dWout[i][j] += dlogits[i] * cache.MlpOut[t][j]
			}
			grad.dbout[i] += dlogits[i]
		}
		// 4.3. dmlpOut = Wout^T * dlogits
		dmlpOut := make([]float64, model.Cfg.EmbedSize)
		for j := 0; j < model.Cfg.EmbedSize; j++ {
			sum := 0.0
			for i := 0; i < model.Cfg.VocabSize; i++ {
				sum += model.Wout[i][j] * dlogits[i]
			}
			dmlpOut[j] = sum
		}
		// 4.4. Backprop через второй слой MLP: mlpOut = W2 * A1 + B2
		for i := 0; i < model.Cfg.EmbedSize; i++ {
			for j := 0; j < model.Cfg.MLPDim; j++ {
				grad.dW2[i][j] += dmlpOut[i] * cache.A1[t][j]
			}
		}
		for i := 0; i < model.Cfg.EmbedSize; i++ {
			grad.db2[i] += dmlpOut[i]
		}
		// 4.5. dReLU = W2^T * dmlpOut
		dReLU := make([]float64, model.Cfg.MLPDim)
		for j := 0; j < model.Cfg.MLPDim; j++ {
			sum := 0.0
			for i := 0; i < model.Cfg.EmbedSize; i++ {
				sum += model.W2[i][j] * dmlpOut[i]
			}
			dReLU[j] = sum
		}
		// 4.6. Backprop через ReLU: dz1 = dReLU * (z1 > 0)
		dz1 := reluBackward(cache.Z1[t], dReLU)
		// 4.7. Градиенты для первого слоя MLP: z1 = W1 * AttentionOut + B1
		for i := 0; i < model.Cfg.MLPDim; i++ {
			for j := 0; j < model.Cfg.EmbedSize; j++ {
				grad.dW1[i][j] += dz1[i] * cache.AttentionOut[t][j]
			}
			grad.db1[i] += dz1[i]
		}
	}
	return grad
}

// =============================================================================
// 5. Оптимизация: Реализация алгоритма Adam для обновления параметров
// =============================================================================

// OptimizerState хранит моменты (m) и квадраты моментов (v) для параметров, необходимые для Adam.
type OptimizerState struct {
	M_W1   [][]float64
	V_W1   [][]float64
	M_B1   []float64
	V_B1   []float64
	M_W2   [][]float64
	V_W2   [][]float64
	M_B2   []float64
	V_B2   []float64
	M_Wout [][]float64
	V_Wout [][]float64
	M_Bout []float64
	V_Bout []float64
	Step   int
}

// initOptimizerState инициализирует состояние оптимизатора (Adam) для параметров модели (только для MLP и финального слоя).
func initOptimizerState(cfg TransformerConfig) *OptimizerState {
	// Инициализируем матрицы нулями
	mW1 := make([][]float64, cfg.MLPDim)
	vW1 := make([][]float64, cfg.MLPDim)
	for i := 0; i < cfg.MLPDim; i++ {
		mW1[i] = make([]float64, cfg.EmbedSize)
		vW1[i] = make([]float64, cfg.EmbedSize)
	}
	mW2 := make([][]float64, cfg.EmbedSize)
	vW2 := make([][]float64, cfg.EmbedSize)
	for i := 0; i < cfg.EmbedSize; i++ {
		mW2[i] = make([]float64, cfg.MLPDim)
		vW2[i] = make([]float64, cfg.MLPDim)
	}
	mWout := make([][]float64, cfg.VocabSize)
	vWout := make([][]float64, cfg.VocabSize)
	for i := 0; i < cfg.VocabSize; i++ {
		mWout[i] = make([]float64, cfg.EmbedSize)
		vWout[i] = make([]float64, cfg.EmbedSize)
	}
	return &OptimizerState{
		M_W1:   mW1,
		V_W1:   vW1,
		M_B1:   make([]float64, cfg.MLPDim),
		V_B1:   make([]float64, cfg.MLPDim),
		M_W2:   mW2,
		V_W2:   vW2,
		M_B2:   make([]float64, cfg.EmbedSize),
		V_B2:   make([]float64, cfg.EmbedSize),
		M_Wout: mWout,
		V_Wout: vWout,
		M_Bout: make([]float64, cfg.VocabSize),
		V_Bout: make([]float64, cfg.VocabSize),
		Step:   0,
	}
}

// adamUpdateParameters обновляет параметры модели (только для MLP и финального слоя) с использованием Adam.
func adamUpdateParameters(model *TransformerModel, grad *Gradients, opt *OptimizerState, beta1, beta2, epsilon float64) {
	opt.Step++
	lr := model.Cfg.LearningRate

	// Обновление для W1 и B1.
	for i := 0; i < len(model.W1); i++ {
		for j := 0; j < len(model.W1[i]); j++ {
			// Обновляем моменты
			opt.M_W1[i][j] = beta1*opt.M_W1[i][j] + (1-beta1)*grad.dW1[i][j]
			opt.V_W1[i][j] = beta2*opt.V_W1[i][j] + (1-beta2)*grad.dW1[i][j]*grad.dW1[i][j]
			// Коррекция смещений
			mHat := opt.M_W1[i][j] / (1 - math.Pow(beta1, float64(opt.Step)))
			vHat := opt.V_W1[i][j] / (1 - math.Pow(beta2, float64(opt.Step)))
			// Обновляем параметр
			model.W1[i][j] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
		}
	}
	for i := 0; i < len(model.B1); i++ {
		opt.M_B1[i] = beta1*opt.M_B1[i] + (1-beta1)*grad.db1[i]
		opt.V_B1[i] = beta2*opt.V_B1[i] + (1-beta2)*grad.db1[i]*grad.db1[i]
		mHat := opt.M_B1[i] / (1 - math.Pow(beta1, float64(opt.Step)))
		vHat := opt.V_B1[i] / (1 - math.Pow(beta2, float64(opt.Step)))
		model.B1[i] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
	// Обновление для W2 и B2.
	for i := 0; i < len(model.W2); i++ {
		for j := 0; j < len(model.W2[i]); j++ {
			opt.M_W2[i][j] = beta1*opt.M_W2[i][j] + (1-beta1)*grad.dW2[i][j]
			opt.V_W2[i][j] = beta2*opt.V_W2[i][j] + (1-beta2)*grad.dW2[i][j]*grad.dW2[i][j]
			mHat := opt.M_W2[i][j] / (1 - math.Pow(beta1, float64(opt.Step)))
			vHat := opt.V_W2[i][j] / (1 - math.Pow(beta2, float64(opt.Step)))
			model.W2[i][j] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
		}
	}
	for i := 0; i < len(model.B2); i++ {
		opt.M_B2[i] = beta1*opt.M_B2[i] + (1-beta1)*grad.db2[i]
		opt.V_B2[i] = beta2*opt.V_B2[i] + (1-beta2)*grad.db2[i]*grad.db2[i]
		mHat := opt.M_B2[i] / (1 - math.Pow(beta1, float64(opt.Step)))
		vHat := opt.V_B2[i] / (1 - math.Pow(beta2, float64(opt.Step)))
		model.B2[i] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
	// Обновление для Wout и Bout.
	for i := 0; i < len(model.Wout); i++ {
		for j := 0; j < len(model.Wout[i]); j++ {
			opt.M_Wout[i][j] = beta1*opt.M_Wout[i][j] + (1-beta1)*grad.dWout[i][j]
			opt.V_Wout[i][j] = beta2*opt.V_Wout[i][j] + (1-beta2)*grad.dWout[i][j]*grad.dWout[i][j]
			mHat := opt.M_Wout[i][j] / (1 - math.Pow(beta1, float64(opt.Step)))
			vHat := opt.V_Wout[i][j] / (1 - math.Pow(beta2, float64(opt.Step)))
			model.Wout[i][j] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
		}
	}
	for i := 0; i < len(model.Bout); i++ {
		opt.M_Bout[i] = beta1*opt.M_Bout[i] + (1-beta1)*grad.dbout[i]
		opt.V_Bout[i] = beta2*opt.V_Bout[i] + (1-beta2)*grad.dbout[i]*grad.dbout[i]
		mHat := opt.M_Bout[i] / (1 - math.Pow(beta1, float64(opt.Step)))
		vHat := opt.V_Bout[i] / (1 - math.Pow(beta2, float64(opt.Step)))
		model.Bout[i] -= lr * mHat / (math.Sqrt(vHat) + epsilon)
	}
}

// =============================================================================
// 6. Сохранение и загрузка модели (через JSON)
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
// 7. Генерация текста (Inference)
// =============================================================================

// generateText генерирует текст посимвольно, начиная с заданных стартовых токенов.
// Если генерируется символ-разделитель (. ! ?), генерация прекращается.
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
		logits, _ := model.forwardWithCache(input)
		if len(logits) == 0 {
			break
		}
		lastLogits := logits[len(logits)-1]
		probs := softmax(lastLogits)
		nextToken := sampleFromDistribution(probs, temperature, topK)
		tokens = append(tokens, nextToken)
		r := idx2char[nextToken]
		if r == '.' || r == '!' || r == '?' {
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

// sampleFromDistribution выбирает следующий токен с учетом температуры и topK.
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
	totalProb := 0.0
	for i := 0; i < topK; i++ {
		totalProb += pairs[i].Prob
	}
	if totalProb == 0 {
		return pairs[0].Index
	}
	r := rand.Float64() * totalProb
	cumSum := 0.0
	for i := 0; i < topK; i++ {
		cumSum += pairs[i].Prob
		if r < cumSum {
			return pairs[i].Index
		}
	}
	return pairs[topK-1].Index
}

// =============================================================================
// 8. Загрузка датасета (TinyShakespeare)
// =============================================================================

// loadDataset читает текстовый файл, строит словарь символов и возвращает последовательность токенов.
func loadDataset(path string) ([]int, map[rune]int, map[int]rune) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		log.Fatalf("Ошибка при чтении файла: %v", err)
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
// 9. Обучение модели
// =============================================================================

// train выполняет цикл обучения: выбирает случайный отрезок, делает forward, вычисляет потерю,
// выполняет backpropagation и обновляет параметры с использованием Adam.
func train(model *TransformerModel, data []int, char2idx map[rune]int, idx2char map[int]rune) {
	cfg := model.Cfg
	N := len(data)
	// Инициализируем состояние оптимизатора Adam для обновляемых параметров.
	optState := initOptimizerState(cfg)
	// Параметры Adam.
	beta1 := 0.9
	beta2 := 0.999
	epsilon := 1e-8

	for iter := 0; iter < cfg.NumIterations; iter++ {
		startIdx := rand.Intn(N - cfg.BlockSize - 1)
		inputSeq := data[startIdx : startIdx+cfg.BlockSize]
		targetSeq := data[startIdx+1 : startIdx+cfg.BlockSize+1]

		logits, cache := model.forwardWithCache(inputSeq)

		// Вычисляем суммарную потерю по всем временным шагам.
		totalLoss := 0.0
		for t := 0; t < len(logits); t++ {
			probs := softmax(logits[t])
			loss := -math.Log(probs[targetSeq[t]] + 1e-9)
			totalLoss += loss
		}
		avgLoss := totalLoss / float64(len(logits))

		if iter%100 == 0 {
			fmt.Printf("Итерация %d, Средняя потеря: %.4f\n", iter, avgLoss)
			generated := generateText(model, inputSeq[:10], 200, idx2char, 0.8, 5)
			fmt.Println("Пример сгенерированного текста:")
			fmt.Println(generated)
			fmt.Println("--------------------------")
		}

		grad := computeGradients(model, inputSeq, targetSeq, cache)
		adamUpdateParameters(model, grad, optState, beta1, beta2, epsilon)
	}
	saveModel(model, "model.json")
}

// =============================================================================
// 10. Главная функция main
// =============================================================================

func main() {
	rand.Seed(time.Now().UnixNano())

	datasetPath := "tinyshakespeare.txt"
	if _, err := os.Stat(datasetPath); os.IsNotExist(err) {
		log.Fatalf("Файл %s не найден. Поместите датасет в рабочую директорию.", datasetPath)
	}

	data, char2idx, idx2char := loadDataset(datasetPath)
	fmt.Printf("Датасет загружен: %d символов, размер словаря: %d\n", len(data), len(char2idx))

	// Настраиваем гиперпараметры (более "серьёзные" параметры).
	cfg := TransformerConfig{
		VocabSize:     len(char2idx),
		EmbedSize:     128,    // Размер эмбеддингов увеличен до 128
		BlockSize:     128,    // Длина входной последовательности 128 символов
		NumHeads:      1,      // Используется одна голова внимания (для простоты)
		MLPDim:        256,    // Размер скрытого слоя MLP увеличен до 256
		LearningRate:  0.0005, // Скорость обучения
		NumIterations: 15000,  // Увеличено число итераций для лучшего обучения
		BatchSize:     64,     // Не используется в данном примере
	}

	var model *TransformerModel
	model, err := loadModel("model.json")
	if err != nil {
		fmt.Println("Сохранённая модель не найдена, создаём новую модель...")
		model = NewTransformerModel(cfg)
	} else {
		fmt.Println("Модель загружена!")
		model.Cfg = cfg
		// Если PositionalEmbeddings пустой, инициализируем его заново.
		if len(model.PositionalEmbeddings) == 0 {
			fmt.Println("Пустой PositionalEmbeddings, инициализируем заново...")
			model.PositionalEmbeddings = newRandomMatrix(cfg.BlockSize, cfg.EmbedSize)
		}
	}

	fmt.Println("Начинается обучение модели...")
	train(model, data, char2idx, idx2char)
	fmt.Println("Обучение завершено!")

	fmt.Println("Генерация финального текста:")
	finalText := generateText(model, []int{0}, 200, idx2char, 0.8, 5)
	fmt.Println(finalText)
}

