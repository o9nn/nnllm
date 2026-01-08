import { describe, it, expect, beforeEach } from 'vitest';
import { EchoStateNetwork } from '../src/neuro-symbolic/echo-state-network.js';

describe('EchoStateNetwork', () => {
  let esn: EchoStateNetwork;

  beforeEach(() => {
    esn = new EchoStateNetwork({
      inputSize: 10,
      reservoirSize: 50,
      outputSize: 5,
      spectralRadius: 0.95,
      leakingRate: 0.3,
      sparsity: 0.1,
    });
  });

  it('should create an ESN with correct configuration', () => {
    expect(esn).toBeDefined();
    expect(esn['inputSize']).toBe(10);
    expect(esn['reservoirSize']).toBe(50);
    expect(esn['outputSize']).toBe(5);
  });

  it('should initialize internal state to zeros', () => {
    const state = esn.getState();
    expect(state).toHaveLength(50);
    expect(state.every((val) => val === 0)).toBe(true);
  });

  it('should update state with valid input', () => {
    const input = new Array(10).fill(0.5);
    esn.update(input);
    const state = esn.getState();

    // State should change after update
    expect(state.some((val) => val !== 0)).toBe(true);
  });

  it('should throw error on input size mismatch', () => {
    const invalidInput = new Array(5).fill(0.5);
    expect(() => esn.update(invalidInput)).toThrow('Input size mismatch');
  });

  it('should produce output vector of correct size', () => {
    const input = new Array(10).fill(0.3);
    esn.update(input);
    const output = esn.getOutput();

    expect(output).toHaveLength(5);
    expect(Array.isArray(output)).toBe(true);
  });

  it('should process a sequence of inputs', () => {
    const sequence = [
      new Array(10).fill(0.1),
      new Array(10).fill(0.3),
      new Array(10).fill(0.5),
    ];

    const outputs = esn.processSequence(sequence);

    expect(outputs).toHaveLength(3);
    expect(outputs[0]).toHaveLength(5);
  });

  it('should reset state to zeros', () => {
    const input = new Array(10).fill(0.5);
    esn.update(input);

    // State should be non-zero
    expect(esn.getState().some((val) => val !== 0)).toBe(true);

    esn.reset();

    // State should be zero after reset
    expect(esn.getState().every((val) => val === 0)).toBe(true);
  });

  it('should encode text into numerical representation', () => {
    const text = 'hello';
    const encoded = EchoStateNetwork.encodeText(text, 10);

    expect(encoded).toHaveLength(10);
    expect(encoded.every((val) => val >= 0 && val <= 1)).toBe(true);
  });

  it('should process text and produce output', () => {
    const text = 'test input';
    const output = esn.processText(text);

    expect(output).toHaveLength(5);
    expect(Array.isArray(output)).toBe(true);
  });

  it('should train output weights', () => {
    const inputs = [
      new Array(10).fill(0.2),
      new Array(10).fill(0.4),
      new Array(10).fill(0.6),
    ];

    const targets = [
      new Array(5).fill(0.1),
      new Array(5).fill(0.2),
      new Array(5).fill(0.3),
    ];

    expect(() => esn.train(inputs, targets)).not.toThrow();
  });

  it('should throw error when training with mismatched inputs and targets', () => {
    const inputs = [new Array(10).fill(0.2)];
    const targets = [new Array(5).fill(0.1), new Array(5).fill(0.2)];

    expect(() => esn.train(inputs, targets)).toThrow(
      'Number of inputs and targets must match',
    );
  });

  it('should maintain echo state property with leaking rate', () => {
    const input = new Array(10).fill(0.8);

    esn.update(input);
    const state1 = [...esn.getState()];

    esn.update(new Array(10).fill(0));
    const state2 = [...esn.getState()];

    // State should decay but not be zero immediately (echo property)
    expect(state2.some((val) => Math.abs(val) > 0.01)).toBe(true);
    expect(state2.some((val) => Math.abs(val) < Math.abs(state1[0]))).toBe(
      true,
    );
  });
});
