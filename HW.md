## 简介

本次训练营将直接使用InfiniCore项目作为作业所用工程，旨在为初学者提供实际开发工作中的寒武纪算子开发体验。
其中跨平台通用的代码以及主要调用流程均已实现，学生可以聚焦寒武纪算子开发的部分。

### 与作业相关的项目结构概览

- `\src\infiniop\ops`：各算子的具体定义及跨平台实现
  - `\src\infiniop\ops\sub`：减法算子
  - `\src\infiniop\ops\causal_softmax`：causal softmax算子
  - `\src\infiniop\ops\random_sample`：random sample算子
  - `\src\infiniop\ops\rearrange`：rearrange算子
- `\src\infiniop\reduce`：各平台通用reduce方法

- `\test\infiniop`：各算子的python测试脚本 - 供参考，勿修改
- `\test\infiniop-test\test_generate\testcases`：各算子的gguf测试脚本 - 供参考，勿修改

## 作业 #0：起步
- 请同学们将工程Fork到自己的账号
- 所有需要的库和环境变量均已配置在镜像中
- 配置寒武纪编译选项--cambricon-mlu=true
- 可以直接编译和运行测试
### 预期结果：
- sub不通过 - 未实现
- mul不通过 - 未实现
- causal softmax不通过 - 调库不支持strides

## 作业 #1：入门 - 实现寒武纪的sub算子
### 预期结果：
1. 可以通过python测试
2. 可以通过gguf测试
3. 性能不明显弱于pytorch实现 (使用--profile测试选项)

## 作业 #2：实现寒武纪的手写causal softmax算子
### 预期结果：
1. 可以通过python测试
2. 可以通过gguf测试
3. 性能优于pytorch实现

## Bonus作业 #1：优化rearrange算子
### 预期结果：
1. 性能优于pytorch实现

## Bonus作业 #2：优化random sample算子
### 预期结果：
1. 性能优于pytorch实现

## 作业提交
对wooway777/InfiniCore Cambricon370z分支提交pr

## 项目介绍

### 项目模块体系

- infini-utils：全模块通用工具代码。
- infinirt：运行时库，依赖 infini-utils。
- infiniop：算子库，依赖 infinirt。除了 C++ 算子实现之外，也包括使用九齿（triton）的算子实现，这部分算子需要在编译之前使用脚本生成源文件。安装后可以运行位于 `test/infiniop` 中的单测脚本进行测试。
- infiniccl：通信库，依赖 infinirt。
- utils-test：工具库测试代码，依赖 infini-utils。
- infiniop-test：算子库测试框架代码。与单测不同，读取gguf测例文件进行测试（详见[`测例文档`](test/infiniop-test/README.md)）。使用前需要安装好 infiniop。
- infiniccl-test：通信库测试代码，使用前需要安装好 infiniccl。

### 文件目录结构

```bash
├── xmake.lua  # 总体 xmake 编译配置，包含所有平台的编译选项和宏定义
├── xmake/*.lua  # 各平台 xmake 编译配置， 包含各平台特有的编译方式
│    
├── include/  # 对外暴露的头文件目录，安装时会被复制到安装目录
│   ├── infiniop/*.h  # InfiniOP算子库子头文件
│   ├── *.h  # 模块核心头文件
│ 
├── src/  # 各模块源代码目录，包含源代码文件以及不对外暴露的头文件
│   ├── infiniop/ # InfiniOP算子库源代码目录
│   │   ├── devices/  # 每个设备平台各自的通用代码目录
│   │   ├── ops/ # 算子实现代码目录
│   │   │   ├── [op]/
│   │   │   │   ├── [device]/ # 各硬件平台算子实现代码目录
│   │   │   │   ├── operator.cc # 算子C语言接口实现
│   │   ├── reduce/ # 规约类算子通用代码目录
│   │   ├── elementwise/  # 逐元素类算子通用代码目录
│   │   ├── *.h  # 核心结构体定义
│   │
│   ├── infiniop-test/  # InfiniOP算子库测试框架
│   ├── infinirt/ # InfiniRT运行时库源代码目录
│   ├── infiniccl/ # InfiniCCL集合通信库源代码目录
│  
├── test/ # 测试源代码目录
│   ├── infiniop/ # InfiniOP算子库单元测试目录
│   │       ├── *.py     # 单测脚本（依赖各平台PyTorch）
│   ├── infiniop-test/
│   │       ├── test_generate/ # 算子库测试框架测例生成脚本
│  
├── scripts/ # 脚本目录
│   ├── install.py # 安装编译脚本
│   ├── python_test.py # 运行所有单测脚本
```

### C++ 代码命名书写规范

1. 类型

    内部数据结构类型 `UpperCamelCase`

    ```c++
    // 尽量使用 Infinixx 开头
    struct InfiniopMatmulCudaDescriptor;
    
    template <typename KeyType, typename ValueType>
    class HashMap; 
    
    using ValueMap = std::unordered_map<int, std::string>;
    ```

    对外暴露的指针类型和枚举类型 `infinixx[XxxXxx]_t`

    常量使用 `INFINI_UPPER_SNAKE_CASE`

    ```c++
    typedef struct InfiniopMatmulCudaDescriptor *infiniopMatmulCudaDescriptor_t;
    
    typedef enum {
        // INFINI...
        INFINI_DTYPE_INVALID = 0,
    } infiniDtype_t;
    ```

2. 普通变量、形参、类数据成员，使用 `snake_case`

    成员名前下划线特指private成员，其他情况应避免使用前下划线

    ```c++
    int max_count;
    
    class Example {
    public:
        std::string getUserName(std::string user_id);
    private:
        // private数据成员名字前加下划线
        int _max_count;
        std::string _user_name;
    };
    
    struct UrlTableProperties {  
        string name;
        int num_entries;  
        static Pool<UrlTableProperties>* pool;
    };
    ```

    当形参与函数内部变量或成员变量重名，可选择其中一个名字后加下划线。当函数内部临时变量和成员重名时，临时变量名字后加下划线。后下划线表示“临时”

    ```c++
    void do(int count_){
        int count = count_;
    }
    ```

3. 函数，使用 lowerCamelCase

    ```c++
    int getMaxValue() const;
    ```

4. const/volatile修饰符写在类型前面

    ```c++
    const void *ptr;
    const int num;
    ```

### 代码格式化

本项目分别使用 `clang-format-16` 和 `black` 对 C/C++ 以及 Python 代码进行格式化。可以使用 [`scripts/format.py`](/scripts/format.py) 脚本实现代码格式化检查和操作。

使用

```shell
python scripts/format.py -h
```

查看脚本帮助信息：

```plaintext
usage: format.py [-h] [--ref REF] [--path [PATH ...]] [--check] [--c C] [--py PY]

options:
  -h, --help         show this help message and exit
  --ref REF          Git reference (commit hash) to compare against.
  --path [PATH ...]  Files to format or check.
  --check            Check files without modifying them.
  --c C              C formatter (default: clang-format-16)
  --py PY            Python formatter (default: black)
```

参数中：

- `ref` 和 `path` 控制格式化的文件范围
  - 若 `ref` 和 `path` 都为空，格式化当前暂存（git added）的文件；
  - 否则
    - 若 `ref` 非空，将比较指定 commit 和当前代码的差异，只格式化修改过的文件；
    - 若 `path` 非空，可传入多个路径（`--path p0 p1 p2`），只格式化指定路径及其子目录中的文件；
- 若设置 `--check`，将检查代码是否需要修改格式，不修改文件内容；
- 通过 `--c` 指定 c/c++ 格式化器，默认为 `clang-format-16`；
- 通过 `--python` 指定 python 格式化器 `black`；
