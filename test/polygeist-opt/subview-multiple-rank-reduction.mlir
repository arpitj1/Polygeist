// RUN: polygeist-opt %s --canonicalize | FileCheck %s

// Dropping several axes is legal. Preserve every fixed coordinate and leave
// the general subview to upstream MemRef lowering.
// CHECK-LABEL: func.func @scalar_view
// CHECK: memref.subview
// CHECK: memref.load
// CHECK: return
func.func @scalar_view(%base:memref<?x3x4xf32>,%row:index) -> f32 {
 %unused = "polygeist.memref2pointer"(%base) : (memref<?x3x4xf32>) -> !llvm.ptr
 %view = memref.subview %base[%row,0,0] [1,1,1] [1,1,1] : memref<?x3x4xf32> to memref<f32,strided<[],offset:?>>
 %value = memref.load %view[] : memref<f32,strided<[],offset:?>>
 return %value : f32
}

// CHECK-LABEL: func.func @vector_view
// CHECK: memref.subview
// CHECK: memref.store
// CHECK: return
func.func @vector_view(%base:memref<?x3x4xf32>,%row:index,%column:index,%value:f32) {
 %view = memref.subview %base[%row,0,0] [1,1,4] [1,1,1] : memref<?x3x4xf32> to memref<4xf32,strided<[1],offset:?>>
 memref.store %value,%view[%column] : memref<4xf32,strided<[1],offset:?>>
 return
}
