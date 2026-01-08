/**
 * OpenCog Neuro-Symbolic LLM Integration
 *
 * This module integrates the echo-state neural network perception layer
 * with the tensor-logic reasoning engine to create a neuro-symbolic AI system
 * inspired by OpenCog architecture.
 */

import { EchoStateNetwork, ESNConfig } from './echo-state-network.js';
import { TensorLogicEngine, Atom, TruthValue } from './tensor-logic-engine.js';

export interface NeuroSymbolicConfig {
  esnConfig?: ESNConfig;
  tensorSize?: number;
  enablePerception?: boolean;
  enableReasoning?: boolean;
}

export interface ProcessingResult {
  perception: number[];
  reasoning: {
    answer: string;
    confidence: number;
    reasoning: string[];
  };
  integration: {
    neuralActivation: number;
    symbolicConfidence: number;
    synergy: number;
  };
}

/**
 * OpenCog-style Neuro-Symbolic Architecture
 * Combines neural perception (ESN) with symbolic reasoning (Tensor-Logic)
 */
export class NeuroSymbolicLLM {
  private perceptionLayer: EchoStateNetwork;
  private reasoningEngine: TensorLogicEngine;
  private enablePerception: boolean;
  private enableReasoning: boolean;
  private processingHistory: ProcessingResult[];

  constructor(config: NeuroSymbolicConfig = {}) {
    // Default ESN configuration for text perception
    const defaultESNConfig: ESNConfig = {
      inputSize: 64, // Character/token embedding size
      reservoirSize: 256, // Reservoir neurons
      outputSize: 128, // Output feature vector
      spectralRadius: 0.95,
      leakingRate: 0.3,
      sparsity: 0.1,
    };

    this.perceptionLayer = new EchoStateNetwork(
      config.esnConfig || defaultESNConfig,
    );

    this.reasoningEngine = new TensorLogicEngine(config.tensorSize || 128);

    this.enablePerception = config.enablePerception ?? true;
    this.enableReasoning = config.enableReasoning ?? true;
    this.processingHistory = [];

    this.initializeKnowledgeBase();
  }

  /**
   * Initialize basic knowledge base with common sense concepts
   */
  private initializeKnowledgeBase(): void {
    // Add basic concepts
    const concepts = [
      'intelligence',
      'learning',
      'reasoning',
      'perception',
      'knowledge',
      'logic',
      'neural',
      'symbolic',
      'language',
      'understanding',
      'thinking',
      'consciousness',
    ];

    const conceptIds: Map<string, string> = new Map();

    for (const concept of concepts) {
      const id = this.reasoningEngine.addConcept(concept);
      conceptIds.set(concept, id);
    }

    // Add some relations
    if (conceptIds.has('neural') && conceptIds.has('perception')) {
      this.reasoningEngine.addRelation(
        'RelatedTo',
        conceptIds.get('neural')!,
        conceptIds.get('perception')!,
      );
    }

    if (conceptIds.has('symbolic') && conceptIds.has('reasoning')) {
      this.reasoningEngine.addRelation(
        'RelatedTo',
        conceptIds.get('symbolic')!,
        conceptIds.get('reasoning')!,
      );
    }

    if (conceptIds.has('neural') && conceptIds.has('symbolic')) {
      this.reasoningEngine.addRelation(
        'ComplementaryTo',
        conceptIds.get('neural')!,
        conceptIds.get('symbolic')!,
      );
    }
  }

  /**
   * Process input through the neuro-symbolic pipeline
   */
  public process(input: string): ProcessingResult {
    let perceptionOutput: number[] = [];
    let neuralActivation = 0;

    // Step 1: Neural Perception Layer (ESN)
    if (this.enablePerception) {
      perceptionOutput = this.perceptionLayer.processText(input);
      neuralActivation = this.calculateActivation(perceptionOutput);
    }

    // Step 2: Symbolic Reasoning Layer (Tensor-Logic)
    let reasoningResult = {
      answer: 'Reasoning disabled',
      confidence: 0,
      reasoning: [] as string[],
    };
    let symbolicConfidence = 0;

    if (this.enableReasoning) {
      reasoningResult = this.reasoningEngine.reason(input);
      symbolicConfidence = reasoningResult.confidence;
    }

    // Step 3: Cognitive Synergy - Integrate neural and symbolic
    const synergy = this.calculateCognitiveSynergy(
      neuralActivation,
      symbolicConfidence,
    );

    const result: ProcessingResult = {
      perception: perceptionOutput,
      reasoning: reasoningResult,
      integration: {
        neuralActivation,
        symbolicConfidence,
        synergy,
      },
    };

    this.processingHistory.push(result);

    return result;
  }

  /**
   * Calculate neural activation strength from perception output
   */
  private calculateActivation(output: number[]): number {
    if (output.length === 0) return 0;

    const sumSquares = output.reduce((sum, val) => sum + val * val, 0);
    return Math.sqrt(sumSquares) / output.length;
  }

  /**
   * Calculate cognitive synergy between neural and symbolic components
   * This represents the "emergent intelligence" from their interaction
   */
  private calculateCognitiveSynergy(
    neuralActivation: number,
    symbolicConfidence: number,
  ): number {
    // Synergy is higher when both components are active and aligned
    const harmonic =
      (2 * neuralActivation * symbolicConfidence) /
      (neuralActivation + symbolicConfidence + 0.001);
    const geometric = Math.sqrt(neuralActivation * symbolicConfidence);

    // Weighted combination
    return 0.6 * harmonic + 0.4 * geometric;
  }

  /**
   * Enhance LLM prompt with neuro-symbolic context
   */
  public enhancePrompt(originalPrompt: string): string {
    const result = this.process(originalPrompt);

    let enhancedPrompt = originalPrompt;

    // Add perception context if available
    if (this.enablePerception && result.perception.length > 0) {
      const perceptionSummary = `[Neural Perception: activation=${result.integration.neuralActivation.toFixed(3)}]`;
      enhancedPrompt = `${perceptionSummary}\n${enhancedPrompt}`;
    }

    // Add reasoning context if available
    if (this.enableReasoning && result.reasoning.confidence > 0.3) {
      const reasoningContext = `[Symbolic Reasoning: ${result.reasoning.answer}]`;
      enhancedPrompt = `${reasoningContext}\n${enhancedPrompt}`;
    }

    // Add synergy indicator
    if (result.integration.synergy > 0.5) {
      const synergyNote = `[Cognitive Synergy: ${result.integration.synergy.toFixed(3)} - High confidence in neuro-symbolic integration]`;
      enhancedPrompt = `${synergyNote}\n${enhancedPrompt}`;
    }

    return enhancedPrompt;
  }

  /**
   * Post-process LLM response with reasoning
   */
  public postProcessResponse(response: string, originalPrompt: string): string {
    if (!this.enableReasoning) {
      return response;
    }

    // Analyze the response
    const reasoningResult = this.reasoningEngine.reason(response);

    // If confidence is low, add a reasoning note
    if (reasoningResult.confidence < 0.5) {
      return `${response}\n\n[Note: Low symbolic confidence (${reasoningResult.confidence.toFixed(2)}). Consider additional verification.]`;
    }

    return response;
  }

  /**
   * Add new knowledge to the symbolic reasoning engine
   */
  public addKnowledge(
    conceptName: string,
    relations?: Array<{ type: string; target: string }>,
  ): void {
    const conceptId = this.reasoningEngine.addConcept(conceptName);

    if (relations) {
      for (const relation of relations) {
        // Find or create target concept
        const atoms = this.reasoningEngine.getAtomSpace().getAllAtoms();
        let targetId = atoms.find((a) => a.name === relation.target)?.id;

        if (!targetId) {
          targetId = this.reasoningEngine.addConcept(relation.target);
        }

        this.reasoningEngine.addRelation(relation.type, conceptId, targetId);
      }
    }
  }

  /**
   * Train the perception layer with examples
   */
  public trainPerception(inputs: string[], expectedOutputs: number[][]): void {
    if (!this.enablePerception) {
      throw new Error('Perception layer is disabled');
    }

    // Convert text inputs to numerical format
    const numericInputs = inputs.map((text) =>
      EchoStateNetwork.encodeText(text, this.perceptionLayer['inputSize']),
    );

    this.perceptionLayer.train(numericInputs, expectedOutputs);
  }

  /**
   * Get processing history for analysis
   */
  public getProcessingHistory(): ProcessingResult[] {
    return this.processingHistory;
  }

  /**
   * Reset the neuro-symbolic system
   */
  public reset(): void {
    this.perceptionLayer.reset();
    this.reasoningEngine.reset();
    this.processingHistory = [];
    this.initializeKnowledgeBase();
  }

  /**
   * Get statistics about the system
   */
  public getStats(): {
    perceptionEnabled: boolean;
    reasoningEnabled: boolean;
    processedCount: number;
    averageSynergy: number;
    knowledgeBaseSize: number;
  } {
    const avgSynergy =
      this.processingHistory.length > 0
        ? this.processingHistory.reduce(
            (sum, r) => sum + r.integration.synergy,
            0,
          ) / this.processingHistory.length
        : 0;

    return {
      perceptionEnabled: this.enablePerception,
      reasoningEnabled: this.enableReasoning,
      processedCount: this.processingHistory.length,
      averageSynergy: avgSynergy,
      knowledgeBaseSize: this.reasoningEngine.getAtomSpace().getAllAtoms()
        .length,
    };
  }

  /**
   * Export knowledge base
   */
  public exportKnowledgeBase(): { atoms: Atom[]; links: any[] } {
    const atomSpace = this.reasoningEngine.getAtomSpace();
    return {
      atoms: atomSpace.getAllAtoms(),
      links: atomSpace.getAllLinks(),
    };
  }
}
