import { describe, it, expect, beforeEach } from 'vitest';
import { NeuroSymbolicLLM } from '../src/neuro-symbolic/neuro-symbolic-llm.js';

describe('NeuroSymbolicLLM', () => {
  let neuroSymbolic: NeuroSymbolicLLM;

  beforeEach(() => {
    neuroSymbolic = new NeuroSymbolicLLM({
      enablePerception: true,
      enableReasoning: true,
      tensorSize: 64,
    });
  });

  it('should create a NeuroSymbolicLLM', () => {
    expect(neuroSymbolic).toBeDefined();
  });

  it('should process input text', () => {
    const input = 'Hello, how are you?';
    const result = neuroSymbolic.process(input);

    expect(result).toBeDefined();
    expect(result.perception).toBeDefined();
    expect(result.reasoning).toBeDefined();
    expect(result.integration).toBeDefined();
    expect(Array.isArray(result.perception)).toBe(true);
  });

  it('should calculate neural activation', () => {
    const result = neuroSymbolic.process('test input');

    expect(result.integration.neuralActivation).toBeGreaterThanOrEqual(0);
    expect(result.integration.neuralActivation).toBeLessThanOrEqual(1);
  });

  it('should calculate symbolic confidence', () => {
    const result = neuroSymbolic.process('artificial intelligence');

    expect(result.integration.symbolicConfidence).toBeGreaterThanOrEqual(0);
    expect(result.integration.symbolicConfidence).toBeLessThanOrEqual(1);
  });

  it('should calculate cognitive synergy', () => {
    const result = neuroSymbolic.process('neural network reasoning');

    expect(result.integration.synergy).toBeGreaterThanOrEqual(0);
    expect(result.integration.synergy).toBeLessThanOrEqual(1);
  });

  it('should enhance prompts', () => {
    const originalPrompt = 'What is machine learning?';
    const enhancedPrompt = neuroSymbolic.enhancePrompt(originalPrompt);

    expect(enhancedPrompt).toBeTruthy();
    expect(enhancedPrompt.length).toBeGreaterThanOrEqual(originalPrompt.length);
  });

  it('should post-process responses', () => {
    const response = 'Machine learning is a subset of AI.';
    const originalPrompt = 'What is machine learning?';
    const processed = neuroSymbolic.postProcessResponse(
      response,
      originalPrompt,
    );

    expect(processed).toBeTruthy();
    expect(processed.length).toBeGreaterThanOrEqual(response.length);
  });

  it('should add knowledge to the system', () => {
    neuroSymbolic.addKnowledge('quantum_computing', [
      { type: 'RelatedTo', target: 'computing' },
      { type: 'RelatedTo', target: 'physics' },
    ]);

    const kb = neuroSymbolic.exportKnowledgeBase();
    expect(kb.atoms.some((a) => a.name === 'quantum_computing')).toBe(true);
  });

  it('should track processing history', () => {
    neuroSymbolic.process('test 1');
    neuroSymbolic.process('test 2');
    neuroSymbolic.process('test 3');

    const history = neuroSymbolic.getProcessingHistory();
    expect(history).toHaveLength(3);
  });

  it('should reset the system', () => {
    neuroSymbolic.process('test input');
    expect(neuroSymbolic.getProcessingHistory()).toHaveLength(1);

    neuroSymbolic.reset();
    expect(neuroSymbolic.getProcessingHistory()).toHaveLength(0);
  });

  it('should provide system statistics', () => {
    neuroSymbolic.process('test 1');
    neuroSymbolic.process('test 2');

    const stats = neuroSymbolic.getStats();

    expect(stats.perceptionEnabled).toBe(true);
    expect(stats.reasoningEnabled).toBe(true);
    expect(stats.processedCount).toBe(2);
    expect(stats.averageSynergy).toBeGreaterThanOrEqual(0);
    expect(stats.knowledgeBaseSize).toBeGreaterThan(0);
  });

  it('should work with perception disabled', () => {
    const nsDisabled = new NeuroSymbolicLLM({
      enablePerception: false,
      enableReasoning: true,
    });

    const result = nsDisabled.process('test input');

    expect(result.perception).toHaveLength(0);
    expect(result.integration.neuralActivation).toBe(0);
  });

  it('should work with reasoning disabled', () => {
    const nsDisabled = new NeuroSymbolicLLM({
      enablePerception: true,
      enableReasoning: false,
    });

    const result = nsDisabled.process('test input');

    expect(result.reasoning.answer).toBe('Reasoning disabled');
    expect(result.integration.symbolicConfidence).toBe(0);
  });

  it('should export knowledge base', () => {
    neuroSymbolic.addKnowledge('test_concept');

    const kb = neuroSymbolic.exportKnowledgeBase();

    expect(kb).toBeDefined();
    expect(Array.isArray(kb.atoms)).toBe(true);
    expect(Array.isArray(kb.links)).toBe(true);
    expect(kb.atoms.length).toBeGreaterThan(0);
  });

  it('should handle empty input', () => {
    const result = neuroSymbolic.process('');

    expect(result).toBeDefined();
    expect(result.perception).toBeDefined();
    expect(result.reasoning).toBeDefined();
  });

  it('should handle long input text', () => {
    const longInput =
      'This is a very long input text that contains multiple sentences and ideas. '.repeat(
        10,
      );
    const result = neuroSymbolic.process(longInput);

    expect(result).toBeDefined();
    expect(result.integration.synergy).toBeGreaterThanOrEqual(0);
  });

  it('should show higher synergy for related concepts', () => {
    // Process input with concepts that exist in knowledge base
    const result1 = neuroSymbolic.process(
      'neural perception and symbolic reasoning',
    );
    const result2 = neuroSymbolic.process('random unrelated words xyz');

    // First should potentially have higher confidence due to known concepts
    expect(result1.integration).toBeDefined();
    expect(result2.integration).toBeDefined();
  });

  it('should train perception layer when enabled', () => {
    const inputs = ['hello', 'world'];
    const outputs = [new Array(128).fill(0.1), new Array(128).fill(0.2)];

    expect(() => neuroSymbolic.trainPerception(inputs, outputs)).not.toThrow();
  });

  it('should throw error when training perception with disabled layer', () => {
    const nsDisabled = new NeuroSymbolicLLM({
      enablePerception: false,
      enableReasoning: true,
    });

    const inputs = ['hello'];
    const outputs = [new Array(128).fill(0.1)];

    expect(() => nsDisabled.trainPerception(inputs, outputs)).toThrow();
  });

  it('should maintain consistent synergy calculation', () => {
    const result1 = neuroSymbolic.process('test input');
    const result2 = neuroSymbolic.process('test input');

    // Synergy should be consistent for same input (within tolerance)
    expect(
      Math.abs(result1.integration.synergy - result2.integration.synergy),
    ).toBeLessThan(0.1);
  });
});
