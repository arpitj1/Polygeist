// RUN: polygeist-opt --linalg-debufferize %s | FileCheck %s

// Captured reads use the current tensor, including an earlier write.
// CHECK-LABEL: func.func @captured_read
// CHECK: %[[INPUT:.*]] = bufferization.to_tensor %arg0
// CHECK: %[[UPDATED:.*]] = tensor.insert {{.*}} into %[[INPUT]]
// CHECK: linalg.generic
// CHECK-SAME: tensor<16xf32>
// CHECK: tensor.extract %[[UPDATED]]
// CHECK-NOT: memref.load
// CHECK: return
func.func @captured_read(%input: memref<8xf32>, %output: memref<16xf32>,
                         %value: f32) {
  %zero = arith.constant 0 : index
  %two = arith.constant 2 : index
  memref.store %value, %input[%zero] : memref<8xf32>
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]} outs(%output : memref<16xf32>) {
  ^bb0(%out: f32):
    %i = linalg.index 0 : index
    %j = arith.divsi %i, %two : index
    %v = memref.load %input[%j] : memref<8xf32>
    linalg.yield %v : f32
  }
  return
}

// A captured read of a concurrently written root cannot become a snapshot.
// CHECK-LABEL: func.func @captured_output
// CHECK: linalg.generic
// CHECK-SAME: memref<16xf32>
// CHECK: memref.load
// CHECK: return
func.func @captured_output(%output: memref<16xf32>) {
  %zero = arith.constant 0 : index
  linalg.generic {indexing_maps = [affine_map<(i) -> (i)>],
                  iterator_types = ["parallel"]} outs(%output : memref<16xf32>) {
  ^bb0(%out: f32):
    %v = memref.load %output[%zero] : memref<16xf32>
    linalg.yield %v : f32
  }
  return
}
