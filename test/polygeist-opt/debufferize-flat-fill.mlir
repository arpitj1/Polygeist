// RUN: polygeist-opt %s --linalg-debufferize="use-multi-root=true" | FileCheck %s

// A flat fill of complete leading rows shares the base tensor state.
// CHECK-LABEL: func.func @complete_rows
// CHECK-NOT: reinterpret_cast
// CHECK: bufferization.to_tensor
// CHECK: linalg.generic
// CHECK: tensor.extract
// CHECK-NOT: reinterpret_cast
func.func @complete_rows(%base: memref<?x2x3xf32>, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %flat = memref.reinterpret_cast %base to offset: [0], sizes: [18], strides: [1] : memref<?x2x3xf32> to memref<18xf32>
  linalg.fill ins(%seed : f32) outs(%flat : memref<18xf32>)
  %value = memref.load %base[%c0, %c0, %c0] : memref<?x2x3xf32>
  return %value : f32
}

// A partial trailing row does not describe a rectangular prefix.
// CHECK-LABEL: func.func @partial_row
// CHECK: memref.reinterpret_cast
// CHECK: linalg.fill
// CHECK: memref.load
func.func @partial_row(%base: memref<?x2x3xf32>, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %flat = memref.reinterpret_cast %base to offset: [0], sizes: [19], strides: [1] : memref<?x2x3xf32> to memref<19xf32>
  linalg.fill ins(%seed : f32) outs(%flat : memref<19xf32>)
  %value = memref.load %base[%c0, %c0, %c0] : memref<?x2x3xf32>
  return %value : f32
}

// Neither a nonzero offset nor holes in the source layout can be discarded.
// CHECK-LABEL: func.func @offset_view
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
func.func @offset_view(%base: memref<?x2x3xf32>, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %flat = memref.reinterpret_cast %base to offset: [1], sizes: [18], strides: [1] : memref<?x2x3xf32> to memref<18xf32, strided<[1], offset: 1>>
  linalg.fill ins(%seed : f32) outs(%flat : memref<18xf32, strided<[1], offset: 1>>)
  %value = memref.load %base[%c0, %c0, %c0] : memref<?x2x3xf32>
  return %value : f32
}

// CHECK-LABEL: func.func @strided_base
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
func.func @strided_base(%base: memref<?x2x3xf32, strided<[12, 3, 1]>>, %seed: f32) -> f32 {
  %c0 = arith.constant 0 : index
  %flat = memref.reinterpret_cast %base to offset: [0], sizes: [18], strides: [1] : memref<?x2x3xf32, strided<[12, 3, 1]>> to memref<18xf32>
  linalg.fill ins(%seed : f32) outs(%flat : memref<18xf32>)
  %value = memref.load %base[%c0, %c0, %c0] : memref<?x2x3xf32, strided<[12, 3, 1]>>
  return %value : f32
}
