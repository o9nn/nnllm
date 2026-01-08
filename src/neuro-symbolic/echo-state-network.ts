/**
 * Echo-State Network (ESN) - Reservoir Computing for Perception
 *
 * This module implements an echo-state neural network for processing
 * sequential and temporal data as a perception layer for the neuro-symbolic LLM.
 * ESNs use a fixed random recurrent network (reservoir) with trainable output weights.
 */

export interface ESNConfig {
  inputSize: number;
  reservoirSize: number;
  outputSize: number;
  spectralRadius?: number;
  leakingRate?: number;
  sparsity?: number;
  inputScaling?: number;
  seed?: number;
}

export class EchoStateNetwork {
  private inputSize: number;
  private reservoirSize: number;
  private outputSize: number;
  private spectralRadius: number;
  private leakingRate: number;
  private sparsity: number;
  private inputScaling: number;

  // Network weights
  private inputWeights: number[][];
  private reservoirWeights: number[][];
  private outputWeights: number[][];

  // Internal state
  private state: number[];

  constructor(config: ESNConfig) {
    this.inputSize = config.inputSize;
    this.reservoirSize = config.reservoirSize;
    this.outputSize = config.outputSize;
    this.spectralRadius = config.spectralRadius ?? 0.95;
    this.leakingRate = config.leakingRate ?? 0.3;
    this.sparsity = config.sparsity ?? 0.1;
    this.inputScaling = config.inputScaling ?? 1.0;

    // Initialize state
    this.state = new Array(this.reservoirSize).fill(0);

    // Initialize weights
    this.inputWeights = this.initializeInputWeights();
    this.reservoirWeights = this.initializeReservoirWeights();
    this.outputWeights = this.initializeOutputWeights();
  }

  /**
   * Initialize input-to-reservoir weights
   */
  private initializeInputWeights(): number[][] {
    const weights: number[][] = [];
    for (let i = 0; i < this.reservoirSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.inputSize; j++) {
        row.push((Math.random() * 2 - 1) * this.inputScaling);
      }
      weights.push(row);
    }
    return weights;
  }

  /**
   * Initialize reservoir weights with spectral radius scaling
   */
  private initializeReservoirWeights(): number[][] {
    const weights: number[][] = [];

    // Create sparse random matrix
    for (let i = 0; i < this.reservoirSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.reservoirSize; j++) {
        if (Math.random() < this.sparsity) {
          row.push(Math.random() * 2 - 1);
        } else {
          row.push(0);
        }
      }
      weights.push(row);
    }

    // Scale by spectral radius (simplified - using max norm)
    const maxNorm = this.calculateMaxNorm(weights);
    if (maxNorm > 0) {
      for (let i = 0; i < this.reservoirSize; i++) {
        for (let j = 0; j < this.reservoirSize; j++) {
          weights[i][j] = (weights[i][j] * this.spectralRadius) / maxNorm;
        }
      }
    }

    return weights;
  }

  /**
   * Initialize output weights
   */
  private initializeOutputWeights(): number[][] {
    const weights: number[][] = [];
    for (let i = 0; i < this.outputSize; i++) {
      const row: number[] = [];
      for (let j = 0; j < this.reservoirSize; j++) {
        row.push(Math.random() * 0.01 - 0.005);
      }
      weights.push(row);
    }
    return weights;
  }

  /**
   * Calculate approximate maximum norm (simplified spectral radius)
   */
  private calculateMaxNorm(matrix: number[][]): number {
    let maxNorm = 0;
    for (const row of matrix) {
      const rowNorm = Math.sqrt(row.reduce((sum, val) => sum + val * val, 0));
      maxNorm = Math.max(maxNorm, rowNorm);
    }
    return maxNorm;
  }

  /**
   * Update internal state with new input
   */
  public update(input: number[]): void {
    if (input.length !== this.inputSize) {
      throw new Error(
        `Input size mismatch: expected ${this.inputSize}, got ${input.length}`,
      );
    }

    const newState = new Array(this.reservoirSize).fill(0);

    // Input contribution
    for (let i = 0; i < this.reservoirSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        newState[i] += this.inputWeights[i][j] * input[j];
      }
    }

    // Recurrent contribution
    for (let i = 0; i < this.reservoirSize; i++) {
      for (let j = 0; j < this.reservoirSize; j++) {
        newState[i] += this.reservoirWeights[i][j] * this.state[j];
      }
    }

    // Apply activation function (tanh) and leaking rate
    for (let i = 0; i < this.reservoirSize; i++) {
      newState[i] = Math.tanh(newState[i]);
      this.state[i] =
        (1 - this.leakingRate) * this.state[i] + this.leakingRate * newState[i];
    }
  }

  /**
   * Get current output based on reservoir state
   */
  public getOutput(): number[] {
    const output = new Array(this.outputSize).fill(0);

    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.reservoirSize; j++) {
        output[i] += this.outputWeights[i][j] * this.state[j];
      }
    }

    return output;
  }

  /**
   * Process a sequence of inputs and return outputs
   */
  public processSequence(inputs: number[][]): number[][] {
    const outputs: number[][] = [];

    for (const input of inputs) {
      this.update(input);
      outputs.push(this.getOutput());
    }

    return outputs;
  }

  /**
   * Get current internal state (for inspection/debugging)
   */
  public getState(): number[] {
    return [...this.state];
  }

  /**
   * Get input size
   */
  public getInputSize(): number {
    return this.inputSize;
  }

  /**
   * Reset the internal state
   */
  public reset(): void {
    this.state = new Array(this.reservoirSize).fill(0);
  }

  /**
   * Train output weights using ridge regression
   * This is a simplified training method for the ESN
   */
  public train(
    inputs: number[][],
    targets: number[][],
    regularization: number = 1e-6,
  ): void {
    if (inputs.length !== targets.length) {
      throw new Error('Number of inputs and targets must match');
    }

    // Collect reservoir states
    const states: number[][] = [];
    this.reset();

    for (const input of inputs) {
      this.update(input);
      states.push([...this.state]);
    }

    // Simplified ridge regression: W_out = Y * X^T * (X * X^T + λI)^-1
    // Where X is reservoir states and Y is targets
    // For simplicity, we'll use a basic least squares approximation

    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.reservoirSize; j++) {
        let numerator = 0;
        let denominator = regularization;

        for (let k = 0; k < states.length; k++) {
          numerator += targets[k][i] * states[k][j];
          denominator += states[k][j] * states[k][j];
        }

        this.outputWeights[i][j] = numerator / denominator;
      }
    }

    this.reset();
  }

  /**
   * Encode text input into numerical representation
   */
  public static encodeText(text: string, size: number): number[] {
    const encoded = new Array(size).fill(0);

    for (let i = 0; i < Math.min(text.length, size); i++) {
      // Simple character encoding: normalize to [0, 1]
      encoded[i] = text.charCodeAt(i) / 255.0;
    }

    return encoded;
  }

  /**
   * Process text through the ESN
   */
  public processText(text: string): number[] {
    const tokens = text.split(' ');
    const inputSize = this.inputSize;

    this.reset();

    for (const token of tokens) {
      const encoded = EchoStateNetwork.encodeText(token, inputSize);
      this.update(encoded);
    }

    return this.getOutput();
  }
}
