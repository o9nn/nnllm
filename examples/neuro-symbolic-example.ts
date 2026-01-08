/**
 * Example: Using the Neuro-Symbolic LLM
 *
 * This example demonstrates how to use the neuro-symbolic AI features
 * including echo-state networks, tensor-logic reasoning, and cognitive synergy.
 */

import {
  NeuroSymbolicLLM,
  EchoStateNetwork,
  TensorLogicEngine,
} from '@electron/llm';

// Example 1: Basic Neuro-Symbolic Processing
function basicExample() {
  console.log('=== Example 1: Basic Neuro-Symbolic Processing ===\n');

  const neuroSymbolic = new NeuroSymbolicLLM({
    enablePerception: true,
    enableReasoning: true,
    tensorSize: 128,
  });

  // Process some inputs
  const inputs = [
    'What is artificial intelligence?',
    'How do neural networks work?',
    'Explain symbolic reasoning',
  ];

  for (const input of inputs) {
    const result = neuroSymbolic.process(input);
    console.log(`Input: ${input}`);
    console.log(
      `Neural Activation: ${result.integration.neuralActivation.toFixed(3)}`,
    );
    console.log(
      `Symbolic Confidence: ${result.integration.symbolicConfidence.toFixed(3)}`,
    );
    console.log(`Cognitive Synergy: ${result.integration.synergy.toFixed(3)}`);
    console.log(`Reasoning: ${result.reasoning.answer}\n`);
  }
}

// Example 2: Echo-State Network Perception
function esnExample() {
  console.log('=== Example 2: Echo-State Network Perception ===\n');

  const esn = new EchoStateNetwork({
    inputSize: 32,
    reservoirSize: 128,
    outputSize: 64,
    spectralRadius: 0.95,
    leakingRate: 0.3,
  });

  // Process a sequence of text
  const text = 'The quick brown fox jumps over the lazy dog';
  const output = esn.processText(text);

  console.log(`Input: ${text}`);
  console.log(`Output dimension: ${output.length}`);
  console.log(
    `Output sample: [${output
      .slice(0, 5)
      .map((v) => v.toFixed(3))
      .join(', ')}...]\n`,
  );

  // Show temporal dynamics
  const tokens = text.split(' ');
  console.log('Token-by-token processing:');
  esn.reset();
  for (const token of tokens) {
    const encoded = EchoStateNetwork.encodeText(token, 32);
    esn.update(encoded);
    const state = esn.getState();
    const activation =
      Math.sqrt(state.reduce((s, v) => s + v * v, 0)) / state.length;
    console.log(`  "${token}": activation = ${activation.toFixed(3)}`);
  }
  console.log();
}

// Example 3: Tensor-Logic Reasoning
function tensorLogicExample() {
  console.log('=== Example 3: Tensor-Logic Reasoning ===\n');

  const engine = new TensorLogicEngine(128);

  // Build a knowledge base
  const ai = engine.addConcept('artificial_intelligence');
  const ml = engine.addConcept('machine_learning');
  const dl = engine.addConcept('deep_learning');
  const nn = engine.addConcept('neural_networks');

  // Add relationships
  engine.addRelation('IncludesLink', ai, ml);
  engine.addRelation('IncludesLink', ml, dl);
  engine.addRelation('UsesLink', dl, nn);

  console.log('Knowledge Base:');
  console.log(`  Concepts: ${engine.getAtomSpace().getAllAtoms().length}`);
  console.log(`  Relations: ${engine.getAtomSpace().getAllLinks().length}\n`);

  // Perform reasoning
  const queries = [
    'What is machine learning?',
    'How are neural networks related to AI?',
    'Explain deep learning',
  ];

  for (const query of queries) {
    const result = engine.reason(query);
    console.log(`Query: ${query}`);
    console.log(`Answer: ${result.answer}`);
    console.log(`Confidence: ${result.confidence.toFixed(3)}`);
    console.log(`Reasoning steps: ${result.reasoning.length}\n`);
  }

  // Logical operations
  console.log('Logical Operations:');
  const tv1 = { strength: 0.8, confidence: 0.9 };
  const tv2 = { strength: 0.6, confidence: 0.8 };

  const andResult = TensorLogicEngine.logicalAnd(tv1, tv2);
  const orResult = TensorLogicEngine.logicalOr(tv1, tv2);
  const notResult = TensorLogicEngine.logicalNot(tv1);

  console.log(`  AND(0.8, 0.6) = ${andResult.strength.toFixed(3)}`);
  console.log(`  OR(0.8, 0.6) = ${orResult.strength.toFixed(3)}`);
  console.log(`  NOT(0.8) = ${notResult.strength.toFixed(3)}\n`);
}

// Example 4: Knowledge Management
function knowledgeManagementExample() {
  console.log('=== Example 4: Knowledge Management ===\n');

  const neuroSymbolic = new NeuroSymbolicLLM();

  // Add domain-specific knowledge
  neuroSymbolic.addKnowledge('quantum_computing', [
    { type: 'RelatedTo', target: 'physics' },
    { type: 'RelatedTo', target: 'computing' },
    { type: 'UsesLink', target: 'quantum_mechanics' },
  ]);

  neuroSymbolic.addKnowledge('quantum_mechanics', [
    { type: 'IsA', target: 'physics' },
    { type: 'HasProperty', target: 'superposition' },
  ]);

  // Export and display knowledge base
  const kb = neuroSymbolic.exportKnowledgeBase();
  console.log(`Knowledge Base:`);
  console.log(`  Total atoms: ${kb.atoms.length}`);
  console.log(`  Total links: ${kb.links.length}`);

  console.log(`\nSample concepts:`);
  for (let i = 0; i < Math.min(5, kb.atoms.length); i++) {
    const atom = kb.atoms[i];
    console.log(
      `  - ${atom.name} (${atom.type}): strength=${atom.truthValue.strength.toFixed(2)}`,
    );
  }
  console.log();
}

// Example 5: Prompt Enhancement
function promptEnhancementExample() {
  console.log('=== Example 5: Prompt Enhancement ===\n');

  const neuroSymbolic = new NeuroSymbolicLLM({
    enablePerception: true,
    enableReasoning: true,
  });

  const originalPrompt = 'Explain the concept of artificial intelligence';

  // First, process to build context
  neuroSymbolic.process('artificial intelligence is important');
  neuroSymbolic.process('AI combines learning and reasoning');

  // Now enhance the prompt
  const enhancedPrompt = neuroSymbolic.enhancePrompt(originalPrompt);

  console.log('Original prompt:');
  console.log(originalPrompt);
  console.log('\nEnhanced prompt:');
  console.log(enhancedPrompt);
  console.log();
}

// Example 6: Statistics and Monitoring
function statisticsExample() {
  console.log('=== Example 6: Statistics and Monitoring ===\n');

  const neuroSymbolic = new NeuroSymbolicLLM();

  // Process multiple inputs
  const inputs = [
    'Hello world',
    'Machine learning basics',
    'Neural network architecture',
    'Symbolic reasoning systems',
    'Cognitive computing',
  ];

  for (const input of inputs) {
    neuroSymbolic.process(input);
  }

  // Get statistics
  const stats = neuroSymbolic.getStats();

  console.log('System Statistics:');
  console.log(`  Perception enabled: ${stats.perceptionEnabled}`);
  console.log(`  Reasoning enabled: ${stats.reasoningEnabled}`);
  console.log(`  Total processed: ${stats.processedCount}`);
  console.log(`  Average synergy: ${stats.averageSynergy.toFixed(3)}`);
  console.log(`  Knowledge base size: ${stats.knowledgeBaseSize} atoms`);

  // Get processing history
  const history = neuroSymbolic.getProcessingHistory();
  console.log(`\nProcessing History (last 3):`);
  for (let i = Math.max(0, history.length - 3); i < history.length; i++) {
    const entry = history[i];
    console.log(
      `  ${i + 1}. Synergy: ${entry.integration.synergy.toFixed(3)}, ` +
        `Neural: ${entry.integration.neuralActivation.toFixed(3)}, ` +
        `Symbolic: ${entry.integration.symbolicConfidence.toFixed(3)}`,
    );
  }
  console.log();
}

// Run all examples
function runAllExamples() {
  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║     Neuro-Symbolic LLM Examples                         ║');
  console.log('╚══════════════════════════════════════════════════════════╝\n');

  basicExample();
  esnExample();
  tensorLogicExample();
  knowledgeManagementExample();
  promptEnhancementExample();
  statisticsExample();

  console.log('╔══════════════════════════════════════════════════════════╗');
  console.log('║     All examples completed!                             ║');
  console.log('╚══════════════════════════════════════════════════════════╝');
}

// Run if this is the main module
if (require.main === module) {
  runAllExamples();
}

export {
  basicExample,
  esnExample,
  tensorLogicExample,
  knowledgeManagementExample,
  promptEnhancementExample,
  statisticsExample,
  runAllExamples,
};
