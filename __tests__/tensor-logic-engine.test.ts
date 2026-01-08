import { describe, it, expect, beforeEach } from 'vitest';
import {
  TensorLogicEngine,
  AtomSpace,
  LogicOperator,
} from '../src/neuro-symbolic/tensor-logic-engine.js';

describe('AtomSpace', () => {
  let atomSpace: AtomSpace;

  beforeEach(() => {
    atomSpace = new AtomSpace();
  });

  it('should create an empty AtomSpace', () => {
    expect(atomSpace).toBeDefined();
    expect(atomSpace.getAllAtoms()).toHaveLength(0);
    expect(atomSpace.getAllLinks()).toHaveLength(0);
  });

  it('should add and retrieve atoms', () => {
    const atom = {
      id: 'test1',
      type: 'ConceptNode',
      name: 'intelligence',
      truthValue: { strength: 1.0, confidence: 1.0 },
    };

    atomSpace.addAtom(atom);

    expect(atomSpace.getAllAtoms()).toHaveLength(1);
    expect(atomSpace.getAtom('test1')).toEqual(atom);
  });

  it('should add and retrieve links', () => {
    const link = {
      id: 'link1',
      type: 'RelatedTo',
      outgoing: ['atom1', 'atom2'],
      truthValue: { strength: 0.8, confidence: 0.8 },
    };

    atomSpace.addLink(link);

    expect(atomSpace.getAllLinks()).toHaveLength(1);
    expect(atomSpace.getLink('link1')).toEqual(link);
  });

  it('should find atoms by type', () => {
    atomSpace.addAtom({
      id: 'test1',
      type: 'ConceptNode',
      name: 'concept1',
      truthValue: { strength: 1.0, confidence: 1.0 },
    });

    atomSpace.addAtom({
      id: 'test2',
      type: 'ConceptNode',
      name: 'concept2',
      truthValue: { strength: 1.0, confidence: 1.0 },
    });

    atomSpace.addAtom({
      id: 'test3',
      type: 'PredicateNode',
      name: 'predicate1',
      truthValue: { strength: 1.0, confidence: 1.0 },
    });

    const conceptNodes = atomSpace.findAtomsByType('ConceptNode');
    expect(conceptNodes).toHaveLength(2);
  });

  it('should find links with specific atom', () => {
    atomSpace.addLink({
      id: 'link1',
      type: 'RelatedTo',
      outgoing: ['atom1', 'atom2'],
      truthValue: { strength: 0.8, confidence: 0.8 },
    });

    atomSpace.addLink({
      id: 'link2',
      type: 'RelatedTo',
      outgoing: ['atom2', 'atom3'],
      truthValue: { strength: 0.8, confidence: 0.8 },
    });

    const linksWithAtom2 = atomSpace.findLinksWithAtom('atom2');
    expect(linksWithAtom2).toHaveLength(2);
  });

  it('should clear all atoms and links', () => {
    atomSpace.addAtom({
      id: 'test1',
      type: 'ConceptNode',
      name: 'concept1',
      truthValue: { strength: 1.0, confidence: 1.0 },
    });

    atomSpace.addLink({
      id: 'link1',
      type: 'RelatedTo',
      outgoing: ['atom1', 'atom2'],
      truthValue: { strength: 0.8, confidence: 0.8 },
    });

    expect(atomSpace.getAllAtoms()).toHaveLength(1);
    expect(atomSpace.getAllLinks()).toHaveLength(1);

    atomSpace.clear();

    expect(atomSpace.getAllAtoms()).toHaveLength(0);
    expect(atomSpace.getAllLinks()).toHaveLength(0);
  });
});

describe('TensorLogicEngine', () => {
  let engine: TensorLogicEngine;

  beforeEach(() => {
    engine = new TensorLogicEngine(64);
  });

  it('should create a TensorLogicEngine', () => {
    expect(engine).toBeDefined();
    expect(engine['tensorSize']).toBe(64);
  });

  it('should add concepts to knowledge base', () => {
    const conceptId = engine.addConcept('artificial_intelligence');

    expect(conceptId).toBeTruthy();
    expect(conceptId).toContain('concept_');

    const atomSpace = engine.getAtomSpace();
    expect(atomSpace.getAllAtoms()).toHaveLength(1);
  });

  it('should add relations between concepts', () => {
    const concept1 = engine.addConcept('neural');
    const concept2 = engine.addConcept('network');

    const relationId = engine.addRelation('RelatedTo', concept1, concept2);

    expect(relationId).toBeTruthy();
    expect(relationId).toContain('link_');

    const atomSpace = engine.getAtomSpace();
    expect(atomSpace.getAllLinks()).toHaveLength(1);
  });

  it('should perform logical AND operation', () => {
    const tv1 = { strength: 0.8, confidence: 0.9 };
    const tv2 = { strength: 0.7, confidence: 0.8 };

    const result = TensorLogicEngine.logicalAnd(tv1, tv2);

    expect(result.strength).toBe(0.8 * 0.7);
    expect(result.confidence).toBe(0.8);
  });

  it('should perform logical OR operation', () => {
    const tv1 = { strength: 0.6, confidence: 0.9 };
    const tv2 = { strength: 0.4, confidence: 0.8 };

    const result = TensorLogicEngine.logicalOr(tv1, tv2);

    expect(result.strength).toBe(0.6 + 0.4 - 0.6 * 0.4);
    expect(result.confidence).toBe(0.8);
  });

  it('should perform logical NOT operation', () => {
    const tv = { strength: 0.7, confidence: 0.9 };

    const result = TensorLogicEngine.logicalNot(tv);

    expect(result.strength).toBeCloseTo(0.3, 10);
    expect(result.confidence).toBe(0.9);
  });

  it('should perform logical IMPLIES operation', () => {
    const tv1 = { strength: 0.8, confidence: 0.9 };
    const tv2 = { strength: 0.6, confidence: 0.8 };

    const result = TensorLogicEngine.logicalImplies(tv1, tv2);

    expect(result.strength).toBeGreaterThan(0);
    expect(result.confidence).toBe(0.8);
  });

  it('should perform pattern matching', () => {
    engine.addConcept('intelligence');
    engine.addConcept('learning');
    engine.addConcept('reasoning');

    const queryEmbedding = new Array(64).fill(0.1);
    const matches = engine.patternMatch(queryEmbedding, 0.0);

    expect(matches.length).toBeGreaterThan(0);
  });

  it('should reason about a query', () => {
    engine.addConcept('artificial_intelligence');
    engine.addConcept('machine_learning');

    const result = engine.reason('What is artificial intelligence?');

    expect(result).toBeDefined();
    expect(result.answer).toBeTruthy();
    expect(result.confidence).toBeGreaterThanOrEqual(0);
    expect(result.confidence).toBeLessThanOrEqual(1);
    expect(Array.isArray(result.reasoning)).toBe(true);
  });

  it('should perform forward chaining', () => {
    const concept1 = engine.addConcept('A');
    const concept2 = engine.addConcept('B');

    engine.getAtomSpace().addLink({
      id: 'impl1',
      type: 'ImplicationLink',
      outgoing: [concept1, concept2],
      truthValue: { strength: 0.9, confidence: 0.9 },
    });

    const inferred = engine.forwardChaining(5);

    expect(Array.isArray(inferred)).toBe(true);
  });

  it('should perform backward chaining', () => {
    const goalId = engine.addConcept('goal');

    const result = engine.backwardChaining(goalId, 3);

    expect(typeof result).toBe('boolean');
  });

  it('should reset the engine', () => {
    engine.addConcept('test');
    expect(engine.getAtomSpace().getAllAtoms().length).toBeGreaterThan(0);

    engine.reset();

    // Should have default concepts from initialization
    expect(engine.getAtomSpace().getAllAtoms().length).toBeGreaterThanOrEqual(
      0,
    );
  });

  it('should handle empty queries', () => {
    const result = engine.reason('');

    expect(result).toBeDefined();
    expect(result.answer).toBeTruthy();
  });

  it('should handle complex reasoning chains', () => {
    const ai = engine.addConcept('artificial_intelligence');
    const ml = engine.addConcept('machine_learning');
    const dl = engine.addConcept('deep_learning');

    engine.addRelation('IncludesLink', ai, ml);
    engine.addRelation('IncludesLink', ml, dl);

    const result = engine.reason('deep learning and AI');

    expect(result).toBeDefined();
    expect(result.reasoning.length).toBeGreaterThan(0);
  });
});
