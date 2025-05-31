对于深度学习来说，其模型计算量往往很大（训练尤甚），但在训练时经常看到GPU利用率打不满的情况，这说明瓶颈在GPU算力以外的其他地方，其中一个最重要的影响因素便是显存带宽

下表列举了几个常用显卡的常规参数

| Model       | Memory (GB) | Memory Bandwidth (GB/sec) | FP32 TFLOPS | FP16 TFLOPS |
| ----------- | ----------- | ------------------------- | ----------- | ----------- |
| A100        | 80          | 2039                      | 19.5        | 312         |
| V100        | 16          | 900                       | 15.7        | 125         |
| A6000       | 48          | 768                       | 38          | 150         |
| RTX 3090 TI | 24          | 1008                      | 40          | 160         |

以RTX 3090TI为例，其显存带宽为1008GB/s，假如我们在这上面做elementwise的float32加法运算，考虑到加法运算需要先把tensor从显存读出，计算出结果后再写回显存，那么其每秒最多只能做`1008GB/4/2=126GB/s`次加法运算，远远达不到其理论上限值40TFLOPS

```python
# X，Y均为float32向量
Y = X + 2
```
当然，真实情况不会这么极端，虽然网络架构（CNN，RNN，Transformers等）有很多种，但算子种类有限，其中矩阵乘法是比较计算密集型的，**elementwise op**是比较偏内存copy/write的，由此会拉低模型的整体TFLOPS，我们优化模型性能的一个很重要的方向便是减少elementwise op的个数已达到减少显存数据read/write的目的, 此时算子融合就显得很重要了，如下介绍来自ChatGPT生成的回答（写得比我好多了，直接贴在这里）：

**Operation fusion** (also called **op fusion**) is a performance optimization technique widely used in deep learning frameworks and compilers. It **combines multiple operations into a single kernel** or computational step to improve efficiency.

---

## 🚀 What Is Operation Fusion?

Instead of launching separate kernels (on CPU/GPU) for each operation, **fusing them** means:

* **Reducing kernel launch overhead**
* **Avoiding intermediate memory writes/reads**
* **Enabling better parallelization and caching**

> ✅ Example:

```python
# Unfused operations
y = x * 2         # kernel 1
z = relu(y)       # kernel 2
```

* Requires **two CUDA kernel launches**
* Intermediate result `y` is **written to memory** after kernel 1, and **read again** in kernel 2

```python
# Fused version
z = fused_mul_relu(x)
```

* **One kernel**
* No intermediate memory write for `y`, just compute and proceed directly to ReLU

🔁 **Same total computation** (still multiply + ReLU), but **much faster** due to lower memory traffic and launch overhead.

---

## 🔧 Types of Operation Fusion

There are several common categories of op fusion used in practice:

### 1. **Elementwise Operation Fusion (Pointwise Fusion)**

* Fuse sequences of elementwise ops like `add`, `mul`, `relu`, `sigmoid`.
* Very common because elementwise ops are easy to chain.
* Example: `y = relu(x * scale + bias)`

### 2. **Activation + Linear Fusion**

* Fuse activation functions directly after linear layers.
* Example: `y = relu(matmul(x, W))` becomes a fused matmul + relu op.

### 3. **Convolution + Bias + Activation**

* Extremely common in CNNs.
* Fuse: `conv2d → bias add → activation (e.g., ReLU)`
* Improves GPU throughput and avoids intermediate buffer storage.

### 4. **Normalization + Activation Fusion**

* Example: `batch_norm + relu`, or `layer_norm + gelu`
* Used in BERT/GPT-style Transformer blocks.

### 5. **Transpose + MatMul Fusion**

* In attention mechanisms, it's common to fuse `transpose → matmul` patterns.
* Helps eliminate extra memory reads/writes.

### 6. **Reshape/View + Elementwise Fusion**

* Sometimes reshapes are just metadata operations; fusing them avoids unnecessary compute.

---

## 🧠 Fusion in Practice (by Framework)

| Framework                           | Fusion Approach                                                   |
| ----------------------------------- | ----------------------------------------------------------------- |
| **TensorFlow XLA**                  | Uses graph-level compiler to fuse compatible ops into HLO kernels |
| **PyTorch (TorchScript + nvFuser)** | Supports runtime or JIT fusion for CUDA kernels                   |
| **ONNX Runtime**                    | Fuses subgraphs during optimization (e.g., conv+bn+relu)          |
| **TVM**                             | Uses scheduling and pattern matching to generate fused kernels    |
| **TensorRT**                        | Performs aggressive layer fusion during model conversion          |

---

## ⚙️ Why Fusion Matters


| Optimization               | How It Helps                                                                          |
| -------------------------- | ------------------------------------------------------------------------------------- |
| **Memory bandwidth**       | Fewer reads/writes of intermediate results to global GPU memory                       |
| **Kernel launch overhead** | Fewer individual kernels → fewer scheduling/launch delays                             |
| **Better data locality**   | More chances to keep data in **registers/shared memory**, avoiding slow global memory |
| **Increased parallelism**  | Enables more efficient GPU thread scheduling and usage of warp execution              |

Note that op fusion does not reduce the amount of computation in terms of math operations (like adds/muls), it reduces the computation overhead.

---

## 🧪 Limitations or Challenges

* **Not all ops are fusible** (e.g., if data dependencies are complex).
* **Numerical accuracy** may slightly vary after fusion due to reordering.
* **Fusing too much** can make debugging/tracing harder.
* Some ops need **shape alignment** or **broadcast constraints** to fuse.

---

## ✅ Summary Table

| Fusion Type              | Example                   |
| ------------------------ | ------------------------- |
| Elementwise              | `relu(x + bias)`          |
| Linear + Activation      | `relu(matmul(x, W))`      |
| Conv + Bias + Activation | `relu(conv2d(x, W) + b)`  |
| Norm + Activation        | `gelu(layer_norm(x))`     |
| Transpose + Matmul       | `matmul(transpose(x), y)` |

---


## 参考链接
- 来自沐神的[transformers-benchmarks](https://github.com/mli/transformers-benchmarks/tree/main)
- 沐神的[视频讲解](https://www.bilibili.com/video/BV1LT411F77M?spm_id_from=333.788.videopod.sections&vd_source=4e29e34ec0b6dc6503fee23f1aa2bfee)，其中详细介绍了上面benchmark脚本的用法，和不同transformer结构的测试结论
