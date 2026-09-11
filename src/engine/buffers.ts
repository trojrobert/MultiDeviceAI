// WebGPU buffer utilities for model upload and readback.

export function alignedSize(bytes: number, alignment = 4): number {
  return Math.ceil(bytes / alignment) * alignment;
}

export function createInitializedBuffer(
  device: GPUDevice,
  source: ArrayBuffer | ArrayBufferView,
  usage: GPUBufferUsageFlags,
  label?: string,
): GPUBuffer {
  const view = ArrayBuffer.isView(source)
    ? new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
    : new Uint8Array(source);
  const buffer = device.createBuffer({
    label,
    size: alignedSize(view.byteLength),
    usage,
    mappedAtCreation: true,
  });
  new Uint8Array(buffer.getMappedRange()).set(view);
  buffer.unmap();
  return buffer;
}

export async function readFloatBuffer(
  device: GPUDevice,
  source: GPUBuffer,
  length: number,
): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: alignedSize(length * 4),
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, length * 4);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapMode.READ);
  const result = Float32Array.from(new Float32Array(staging.getMappedRange(), 0, length));
  staging.unmap();
  staging.destroy();
  return result;
}

export function destroyBuffers(buffers: Iterable<GPUBuffer | undefined>): void {
  for (const buffer of buffers) buffer?.destroy();
}
