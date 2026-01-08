/**
 * Tensor-Logic Reasoning Engine (Agent-Zero Style)
 *
 * This module implements a tensor-based logical reasoning system that operates
 * on multi-dimensional knowledge representations. It combines symbolic logic
 * with tensor operations for efficient inference and reasoning.
 */

export enum LogicOperator {
  AND = 'AND',
  OR = 'OR',
  NOT = 'NOT',
  IMPLIES = 'IMPLIES',
  EQUIVALENT = 'EQUIVALENT',
}

export interface Atom {
  id: string;
  type: string;
  name: string;
  truthValue: TruthValue;
  embedding?: number[];
}

export interface TruthValue {
  strength: number; // [0, 1] - degree of belief
  confidence: number; // [0, 1] - confidence in the belief
}

export interface Link {
  id: string;
  type: string;
  outgoing: string[]; // IDs of atoms
  truthValue: TruthValue;
}

export interface InferenceRule {
  name: string;
  premises: string[];
  conclusion: string;
  strength: number;
}

/**
 * AtomSpace - Knowledge representation inspired by OpenCog
 */
export class AtomSpace {
  private atoms: Map<string, Atom>;
  private links: Map<string, Link>;
  private embeddings: Map<string, number[]>;

  constructor() {
    this.atoms = new Map();
    this.links = new Map();
    this.embeddings = new Map();
  }

  /**
   * Add an atom to the AtomSpace
   */
  public addAtom(atom: Atom): void {
    this.atoms.set(atom.id, atom);
    if (atom.embedding) {
      this.embeddings.set(atom.id, atom.embedding);
    }
  }

  /**
   * Add a link to the AtomSpace
   */
  public addLink(link: Link): void {
    this.links.set(link.id, link);
  }

  /**
   * Get atom by ID
   */
  public getAtom(id: string): Atom | undefined {
    return this.atoms.get(id);
  }

  /**
   * Get link by ID
   */
  public getLink(id: string): Link | undefined {
    return this.links.get(id);
  }

  /**
   * Find atoms by type
   */
  public findAtomsByType(type: string): Atom[] {
    return Array.from(this.atoms.values()).filter((atom) => atom.type === type);
  }

  /**
   * Find links involving specific atoms
   */
  public findLinksWithAtom(atomId: string): Link[] {
    return Array.from(this.links.values()).filter((link) =>
      link.outgoing.includes(atomId),
    );
  }

  /**
   * Clear all atoms and links
   */
  public clear(): void {
    this.atoms.clear();
    this.links.clear();
    this.embeddings.clear();
  }

  /**
   * Get all atoms
   */
  public getAllAtoms(): Atom[] {
    return Array.from(this.atoms.values());
  }

  /**
   * Get all links
   */
  public getAllLinks(): Link[] {
    return Array.from(this.links.values());
  }
}

/**
 * Tensor-Logic Reasoning Engine
 */
export class TensorLogicEngine {
  private atomSpace: AtomSpace;
  private inferenceRules: InferenceRule[];
  private tensorSize: number;

  constructor(tensorSize: number = 128) {
    this.atomSpace = new AtomSpace();
    this.inferenceRules = [];
    this.tensorSize = tensorSize;
    this.initializeDefaultRules();
  }

  /**
   * Initialize default inference rules
   */
  private initializeDefaultRules(): void {
    // Modus Ponens: (A → B) ∧ A ⇒ B
    this.inferenceRules.push({
      name: 'modus_ponens',
      premises: ['implies', 'antecedent'],
      conclusion: 'consequent',
      strength: 0.9,
    });

    // Modus Tollens: (A → B) ∧ ¬B ⇒ ¬A
    this.inferenceRules.push({
      name: 'modus_tollens',
      premises: ['implies', 'not_consequent'],
      conclusion: 'not_antecedent',
      strength: 0.85,
    });

    // Conjunction: A ∧ B ⇒ true if both true
    this.inferenceRules.push({
      name: 'conjunction',
      premises: ['premise1', 'premise2'],
      conclusion: 'conjunction',
      strength: 0.95,
    });
  }

  /**
   * Add a concept to the knowledge base
   */
  public addConcept(name: string, embedding?: number[]): string {
    const id = `concept_${Date.now()}_${Math.random()}`;
    const atom: Atom = {
      id,
      type: 'ConceptNode',
      name,
      truthValue: { strength: 1.0, confidence: 1.0 },
      embedding: embedding || this.generateRandomEmbedding(),
    };
    this.atomSpace.addAtom(atom);
    return id;
  }

  /**
   * Add a relation between concepts
   */
  public addRelation(
    relationName: string,
    sourceId: string,
    targetId: string,
  ): string {
    const id = `link_${Date.now()}_${Math.random()}`;
    const link: Link = {
      id,
      type: relationName,
      outgoing: [sourceId, targetId],
      truthValue: { strength: 0.8, confidence: 0.8 },
    };
    this.atomSpace.addLink(link);
    return id;
  }

  /**
   * Perform logical AND operation on truth values
   */
  public static logicalAnd(tv1: TruthValue, tv2: TruthValue): TruthValue {
    return {
      strength: tv1.strength * tv2.strength,
      confidence: Math.min(tv1.confidence, tv2.confidence),
    };
  }

  /**
   * Perform logical OR operation on truth values
   */
  public static logicalOr(tv1: TruthValue, tv2: TruthValue): TruthValue {
    return {
      strength: tv1.strength + tv2.strength - tv1.strength * tv2.strength,
      confidence: Math.min(tv1.confidence, tv2.confidence),
    };
  }

  /**
   * Perform logical NOT operation on truth value
   */
  public static logicalNot(tv: TruthValue): TruthValue {
    return {
      strength: 1 - tv.strength,
      confidence: tv.confidence,
    };
  }

  /**
   * Perform logical IMPLIES operation on truth values
   */
  public static logicalImplies(tv1: TruthValue, tv2: TruthValue): TruthValue {
    // A → B is equivalent to ¬A ∨ B
    const notTv1 = TensorLogicEngine.logicalNot(tv1);
    return TensorLogicEngine.logicalOr(notTv1, tv2);
  }

  /**
   * Calculate similarity between two embeddings
   */
  private cosineSimilarity(emb1: number[], emb2: number[]): number {
    if (emb1.length !== emb2.length) {
      throw new Error('Embeddings must have same dimension');
    }

    let dotProduct = 0;
    let norm1 = 0;
    let norm2 = 0;

    for (let i = 0; i < emb1.length; i++) {
      dotProduct += emb1[i] * emb2[i];
      norm1 += emb1[i] * emb1[i];
      norm2 += emb2[i] * emb2[i];
    }

    const denominator = Math.sqrt(norm1) * Math.sqrt(norm2);
    return denominator > 0 ? dotProduct / denominator : 0;
  }

  /**
   * Generate random embedding vector
   */
  private generateRandomEmbedding(): number[] {
    const embedding = new Array(this.tensorSize);
    for (let i = 0; i < this.tensorSize; i++) {
      embedding[i] = Math.random() * 2 - 1;
    }
    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return embedding.map((val) => val / norm);
  }

  /**
   * Perform pattern matching to find similar concepts
   */
  public patternMatch(
    queryEmbedding: number[],
    threshold: number = 0.7,
  ): Atom[] {
    const matches: Atom[] = [];

    for (const atom of this.atomSpace.getAllAtoms()) {
      if (atom.embedding) {
        const similarity = this.cosineSimilarity(
          queryEmbedding,
          atom.embedding,
        );
        if (similarity >= threshold) {
          matches.push(atom);
        }
      }
    }

    return matches.sort((a, b) => {
      const simA = this.cosineSimilarity(queryEmbedding, a.embedding!);
      const simB = this.cosineSimilarity(queryEmbedding, b.embedding!);
      return simB - simA;
    });
  }

  /**
   * Perform forward chaining inference
   */
  public forwardChaining(maxSteps: number = 10): Atom[] {
    const inferredAtoms: Atom[] = [];
    let step = 0;

    while (step < maxSteps) {
      let inferredSomething = false;

      // Apply modus ponens
      for (const link of this.atomSpace.getAllLinks()) {
        if (link.type === 'ImplicationLink' && link.outgoing.length === 2) {
          const antecedent = this.atomSpace.getAtom(link.outgoing[0]);
          const consequent = this.atomSpace.getAtom(link.outgoing[1]);

          if (antecedent && consequent) {
            // If antecedent is strong enough, strengthen consequent
            if (antecedent.truthValue.strength > 0.7) {
              const newStrength = Math.min(
                1.0,
                consequent.truthValue.strength +
                  0.1 * antecedent.truthValue.strength,
              );

              if (newStrength > consequent.truthValue.strength) {
                consequent.truthValue.strength = newStrength;
                inferredAtoms.push(consequent);
                inferredSomething = true;
              }
            }
          }
        }
      }

      if (!inferredSomething) {
        break;
      }

      step++;
    }

    return inferredAtoms;
  }

  /**
   * Perform backward chaining to prove a goal
   */
  public backwardChaining(goalId: string, maxDepth: number = 5): boolean {
    const goal = this.atomSpace.getAtom(goalId);
    if (!goal) {
      return false;
    }

    // If goal already has high truth value, it's proven
    if (goal.truthValue.strength > 0.9) {
      return true;
    }

    // Find implications that conclude this goal
    const implications = this.atomSpace
      .getAllLinks()
      .filter(
        (link) =>
          link.type === 'ImplicationLink' && link.outgoing[1] === goalId,
      );

    for (const implication of implications) {
      const antecedentId = implication.outgoing[0];

      if (maxDepth > 0) {
        // Recursively try to prove antecedent
        if (this.backwardChaining(antecedentId, maxDepth - 1)) {
          // Update goal truth value
          const antecedent = this.atomSpace.getAtom(antecedentId);
          if (antecedent) {
            goal.truthValue.strength = Math.max(
              goal.truthValue.strength,
              antecedent.truthValue.strength * implication.truthValue.strength,
            );

            if (goal.truthValue.strength > 0.9) {
              return true;
            }
          }
        }
      }
    }

    return goal.truthValue.strength > 0.9;
  }

  /**
   * Reason about a query using tensor operations
   */
  public reason(query: string): {
    answer: string;
    confidence: number;
    reasoning: string[];
  } {
    const reasoning: string[] = [];

    // Encode query into tensor space
    const queryEmbedding = this.encodeQuery(query);
    reasoning.push(
      `Encoded query into ${this.tensorSize}-dimensional tensor space`,
    );

    // Find relevant concepts using pattern matching
    const matches = this.patternMatch(queryEmbedding, 0.5);
    reasoning.push(
      `Found ${matches.length} relevant concepts through pattern matching`,
    );

    // Perform forward chaining to derive new knowledge
    const inferred = this.forwardChaining(5);
    reasoning.push(
      `Inferred ${inferred.length} new facts through forward chaining`,
    );

    // Calculate confidence based on matches and inference
    const confidence =
      matches.length > 0
        ? matches[0].truthValue.strength * matches[0].truthValue.confidence
        : 0.5;

    // Generate answer based on top matches
    const answer =
      matches.length > 0
        ? `Based on reasoning, found concept: ${matches[0].name} (confidence: ${confidence.toFixed(2)})`
        : 'No strong conclusions from reasoning';

    reasoning.push(`Generated answer with confidence ${confidence.toFixed(2)}`);

    return { answer, confidence, reasoning };
  }

  /**
   * Encode a text query into tensor representation
   */
  private encodeQuery(query: string): number[] {
    const embedding = new Array(this.tensorSize).fill(0);

    // Simple encoding: hash words into embedding space
    const words = query.toLowerCase().split(/\s+/);
    for (const word of words) {
      const hash = this.hashString(word);
      for (let i = 0; i < this.tensorSize; i++) {
        embedding[i] += Math.sin(hash + i) * 0.1;
      }
    }

    // Normalize
    const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
    return norm > 0 ? embedding.map((val) => val / norm) : embedding;
  }

  /**
   * Simple string hash function
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      hash = (hash << 5) - hash + str.charCodeAt(i);
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Get the AtomSpace for external access
   */
  public getAtomSpace(): AtomSpace {
    return this.atomSpace;
  }

  /**
   * Reset the reasoning engine
   */
  public reset(): void {
    this.atomSpace.clear();
    this.inferenceRules = [];
    this.initializeDefaultRules();
  }
}
