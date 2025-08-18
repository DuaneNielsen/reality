#include <Python.h>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "consts.hpp"

extern "C" {

// DLPack structures based on official DLPack specification
// https://github.com/dmlc/dlpack/blob/main/include/dlpack/dlpack.h

typedef struct {
    int32_t device_type;  // 1=CPU, 2=CUDA, etc.
    int32_t device_id;    // Device index
} DLDevice;

typedef struct {
    uint8_t code;     // 0=Int, 1=UInt, 2=Float, etc.
    uint8_t bits;     // 8, 16, 32, 64
    uint16_t lanes;   // Always 1 for scalars
} DLDataType;

typedef struct {
    void* data;           // Pointer to tensor data
    DLDevice device;      // Device information
    int ndim;            // Number of dimensions
    DLDataType dtype;    // Data type
    int64_t* shape;      // Shape array (malloc'd)
    int64_t* strides;    // Strides array (NULL = C-contiguous)
    uint64_t byte_offset; // Byte offset (usually 0)
} DLTensor;

typedef struct DLManagedTensor {
    DLTensor dl_tensor;
    void* manager_ctx;    // Context for cleanup
    void (*deleter)(struct DLManagedTensor* self);
} DLManagedTensor;

// Madrona tensor element type constants (from C API)
enum MER_TensorElementType {
    MER_TENSOR_TYPE_UINT8 = 0,
    MER_TENSOR_TYPE_INT8 = 1,
    MER_TENSOR_TYPE_INT16 = 2,
    MER_TENSOR_TYPE_INT32 = 3,
    MER_TENSOR_TYPE_INT64 = 4,
    MER_TENSOR_TYPE_FLOAT16 = 5,
    MER_TENSOR_TYPE_FLOAT32 = 6,
};

// Convert Madrona tensor type to DLPack data type
DLDataType madrona_to_dlpack_dtype(int madrona_type) {
    switch (madrona_type) {
        case MER_TENSOR_TYPE_UINT8:
            return {1, 8, 1};   // UInt8
        case MER_TENSOR_TYPE_INT8:
            return {0, 8, 1};   // Int8
        case MER_TENSOR_TYPE_INT16:
            return {0, 16, 1};  // Int16
        case MER_TENSOR_TYPE_INT32:
            return {0, 32, 1};  // Int32
        case MER_TENSOR_TYPE_INT64:
            return {0, consts::viewer::dlpackInt64Bits, 1};  // Int64
        case MER_TENSOR_TYPE_FLOAT16:
            return {2, 16, 1};  // Float16
        case MER_TENSOR_TYPE_FLOAT32:
            return {2, 32, 1};  // Float32
        default:
            return {2, 32, 1};  // Default to Float32
    }
}

// Deleter function for DLManagedTensor
// IMPORTANT: Only frees DLPack metadata, NOT the actual tensor data
// The tensor data is owned by Madrona and must not be freed here
void dlpack_deleter(DLManagedTensor* self) {
    if (self == nullptr) return;
    
    // Free the shape array (we allocated this)
    if (self->dl_tensor.shape) {
        free(self->dl_tensor.shape);
        self->dl_tensor.shape = nullptr;
    }
    
    // Free the strides array if it exists (we might allocate this in the future)
    if (self->dl_tensor.strides) {
        free(self->dl_tensor.strides);
        self->dl_tensor.strides = nullptr;
    }
    
    // Free the DLManagedTensor struct itself
    free(self);
    
    // NOTE: We deliberately do NOT free self->dl_tensor.data
    // That memory is owned by Madrona and managed by the CFFI layer
}

// Create a DLPack capsule from tensor parameters
PyObject* create_dlpack_capsule(PyObject* self, PyObject* args) {
    // Parse arguments: data_ptr, shape_list, dtype, device_type, device_id
    unsigned long long data_ptr;
    PyObject* shape_list;
    int dtype;
    int device_type;
    int device_id;
    
    if (!PyArg_ParseTuple(args, "KOiii", &data_ptr, &shape_list, &dtype, 
                          &device_type, &device_id)) {
        return nullptr;
    }
    
    // Validate arguments
    if (!PyList_Check(shape_list)) {
        PyErr_SetString(PyExc_TypeError, "Shape must be a list");
        return nullptr;
    }
    
    // Get number of dimensions
    int ndim = PyList_Size(shape_list);
    if (ndim <= 0) {
        PyErr_SetString(PyExc_ValueError, "Shape must have at least one dimension");
        return nullptr;
    }
    
    // Allocate and populate shape array
    int64_t* shape = (int64_t*)malloc(ndim * sizeof(int64_t));
    if (!shape) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate shape array");
        return nullptr;
    }
    
    for (int i = 0; i < ndim; i++) {
        PyObject* dim_obj = PyList_GetItem(shape_list, i);
        if (!PyLong_Check(dim_obj)) {
            free(shape);
            PyErr_SetString(PyExc_TypeError, "Shape dimensions must be integers");
            return nullptr;
        }
        shape[i] = PyLong_AsLongLong(dim_obj);
        if (shape[i] <= 0) {
            free(shape);
            PyErr_SetString(PyExc_ValueError, "Shape dimensions must be positive");
            return nullptr;
        }
    }
    
    // Allocate DLManagedTensor
    DLManagedTensor* managed = (DLManagedTensor*)malloc(sizeof(DLManagedTensor));
    if (!managed) {
        free(shape);
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate DLManagedTensor");
        return nullptr;
    }
    
    // Fill in DLTensor structure
    managed->dl_tensor.data = (void*)data_ptr;
    managed->dl_tensor.device.device_type = device_type;
    managed->dl_tensor.device.device_id = device_id;
    managed->dl_tensor.ndim = ndim;
    managed->dl_tensor.dtype = madrona_to_dlpack_dtype(dtype);
    managed->dl_tensor.shape = shape;
    managed->dl_tensor.strides = nullptr;  // C-contiguous
    managed->dl_tensor.byte_offset = 0;
    
    // Set up management
    managed->manager_ctx = nullptr;  // We don't need additional context
    managed->deleter = dlpack_deleter;
    
    // Create Python capsule with the standard DLPack name
    return PyCapsule_New(managed, "dltensor", nullptr);
}

// Get device information tuple for __dlpack_device__
PyObject* get_dlpack_device(PyObject* self, PyObject* args) {
    int device_type;
    int device_id;
    
    if (!PyArg_ParseTuple(args, "ii", &device_type, &device_id)) {
        return nullptr;
    }
    
    // Return a tuple (device_type, device_id)
    return Py_BuildValue("(ii)", device_type, device_id);
}

// Module method definitions
static PyMethodDef dlpack_methods[] = {
    {"create_dlpack_capsule", create_dlpack_capsule, METH_VARARGS,
     "Create a DLPack capsule from tensor parameters\n"
     "Args: data_ptr (int), shape (list), dtype (int), device_type (int), device_id (int)\n"
     "Returns: PyCapsule with DLManagedTensor"},
    
    {"get_dlpack_device", get_dlpack_device, METH_VARARGS,
     "Get device information tuple\n"
     "Args: device_type (int), device_id (int)\n"
     "Returns: tuple (device_type, device_id)"},
    
    {nullptr, nullptr, 0, nullptr}  // Sentinel
};

// Module definition
static struct PyModuleDef dlpack_module = {
    PyModuleDef_HEAD_INIT,
    "_madrona_escape_room_dlpack",  // Module name
    "DLPack support for Madrona Escape Room tensors",  // Module docstring
    -1,  // Size of per-interpreter state (-1 = global state)
    dlpack_methods
};

// Module initialization function
PyMODINIT_FUNC PyInit__madrona_escape_room_dlpack(void) {
    PyObject* module = PyModule_Create(&dlpack_module);
    if (!module) {
        return nullptr;
    }
    
    // Add module constants
    PyModule_AddIntConstant(module, "DLPACK_VERSION_MAJOR", 0);
    PyModule_AddIntConstant(module, "DLPACK_VERSION_MINOR", consts::viewer::dlpackVersionMinor);
    
    // Device type constants
    PyModule_AddIntConstant(module, "kDLCPU", 1);
    PyModule_AddIntConstant(module, "kDLCUDA", 2);
    
    // Data type constants
    PyModule_AddIntConstant(module, "kDLInt", 0);
    PyModule_AddIntConstant(module, "kDLUInt", 1);
    PyModule_AddIntConstant(module, "kDLFloat", 2);
    
    return module;
}

} // extern "C"