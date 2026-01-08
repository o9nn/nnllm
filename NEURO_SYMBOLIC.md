# OpenCog Neuro-Symbolic LLM

This module implements a neuro-symbolic AI architecture inspired by OpenCog, combining neural perception with symbolic reasoning for enhanced language model capabilities.

## Architecture Overview

The implementation consists of three main components:

### 1. Echo-State Network (ESN) - Neural Perception Layer

The ESN provides reservoir computing capabilities for processing sequential and temporal data. Key features:

- **Reservoir Computing**: Fixed random recurrent network with trainable output weights
- **Dynamic State**: Maintains internal state that captures temporal dependencies
- **Leaking Rate**: Controls memory decay for adaptive processing
- **Spectral Radius**: Ensures echo state property for stable dynamics

**Usage Example:**

```typescript
import { EchoStateNetwork } from '@electron/llm';

const esn = new EchoStateNetwork({
  inputSize: 64,
  reservoirSize: 256,
  outputSize: 128,
  spectralRadius: 0.95,
  leakingRate: 0.3,
});

// Process text input
const output = esn.processText('Hello world');
console.log('Perception output:', output);
```

### 2. Tensor-Logic Reasoning Engine - Symbolic Reasoning Layer

The tensor-logic engine implements symbolic AI capabilities with tensor-based operations. Key features:

- **AtomSpace**: Knowledge graph for storing concepts and relationships
- **Truth Values**: Probabilistic logic with strength and confidence
- **Pattern Matching**: Find similar concepts using tensor embeddings
- **Inference**: Forward and backward chaining for logical reasoning

**Usage Example:**

```typescript
import { TensorLogicEngine } from '@electron/llm';

const engine = new TensorLogicEngine(128);

// Add concepts
const aiId = engine.addConcept('artificial_intelligence');
const mlId = engine.addConcept('machine_learning');

// Add relationship
engine.addRelation('IncludesLink', aiId, mlId);

// Perform reasoning
const result = engine.reason('What is machine learning?');
console.log('Reasoning result:', result);
```

### 3. NeuroSymbolicLLM - Integration Layer

Combines neural perception with symbolic reasoning to create cognitive synergy. Key features:

- **Cognitive Synergy**: Emergent intelligence from neural-symbolic interaction
- **Prompt Enhancement**: Enriches LLM prompts with perception and reasoning context
- **Response Post-Processing**: Validates and enhances LLM responses
- **Knowledge Management**: Dynamic knowledge base with concept learning

**Usage Example:**

```typescript
import { NeuroSymbolicLLM } from '@electron/llm';

const neuroSymbolic = new NeuroSymbolicLLM({
  enablePerception: true,
  enableReasoning: true,
  tensorSize: 128,
});

// Process input through neuro-symbolic pipeline
const result = neuroSymbolic.process('Tell me about neural networks');

console.log('Neural activation:', result.integration.neuralActivation);
console.log('Symbolic confidence:', result.integration.symbolicConfidence);
console.log('Cognitive synergy:', result.integration.synergy);

// Enhance LLM prompt
const enhancedPrompt = neuroSymbolic.enhancePrompt('What is AI?');

// Add new knowledge
neuroSymbolic.addKnowledge('deep_learning', [
  { type: 'IsA', target: 'machine_learning' },
  { type: 'Uses', target: 'neural_networks' },
]);
```

## Integration with Language Model

The neuro-symbolic components are automatically integrated into the main `LanguageModel` class:

```typescript
import { LanguageModel } from '@electron/llm';

// When you create a language model, neuro-symbolic processing is enabled by default
const model = await LanguageModel.create({
  modelPath: '/path/to/model.gguf',
  modelAlias: 'my-model',
});

// Prompts are automatically enhanced with neuro-symbolic context
const response = await model.prompt({
  role: LanguageModelPromptRole.USER,
  type: LanguageModelPromptType.TEXT,
  content: 'Explain quantum computing',
});

// Response includes neuro-symbolic post-processing
console.log(response);
```

## Key Concepts

### Echo State Property

The ESN maintains the "echo state property" where the current state depends on the input history but not the initial conditions. This is achieved through:

- Spectral radius < 1 for stability
- Leaking rate for controllable memory
- Sparse connectivity for efficiency

### Truth Values

Symbolic reasoning uses probabilistic logic with:

- **Strength**: Degree of belief (0-1)
- **Confidence**: Confidence in the belief (0-1)

This allows handling uncertain and incomplete knowledge.

### Cognitive Synergy

The integration layer calculates cognitive synergy as a measure of how well the neural and symbolic components work together:

```
synergy = 0.6 * harmonic_mean + 0.4 * geometric_mean
```

Higher synergy indicates stronger alignment between perception and reasoning.

## Performance Considerations

### ESN Configuration

- **Input Size**: Match your tokenization strategy (64-128 typical)
- **Reservoir Size**: Larger = more memory, slower (256-512 typical)
- **Output Size**: Match tensor size for integration (128 typical)
- **Sparsity**: Higher = faster, less expressive (0.1-0.2 typical)

### Tensor-Logic Configuration

- **Tensor Size**: Dimensionality of embeddings (64-256 typical)
- **Knowledge Base**: Keep under 10,000 atoms for real-time performance
- **Pattern Matching**: Use thresholds > 0.5 for quality matches

## Testing

The implementation includes comprehensive test coverage:

```bash
# Run all tests
npm test

# Run specific test suites
npm test echo-state-network
npm test tensor-logic-engine
npm test neuro-symbolic-llm
```

Test coverage:
- Echo-State Network: 12 tests
- Tensor-Logic Engine: 20 tests
- Neuro-Symbolic Integration: 20 tests

## Advanced Features

### Training the Perception Layer

```typescript
const inputs = ['hello', 'world', 'test'];
const expectedOutputs = [
  new Array(128).fill(0.1),
  new Array(128).fill(0.2),
  new Array(128).fill(0.3),
];

neuroSymbolic.trainPerception(inputs, expectedOutputs);
```

### Exporting Knowledge Base

```typescript
const kb = neuroSymbolic.exportKnowledgeBase();
console.log('Atoms:', kb.atoms.length);
console.log('Links:', kb.links.length);

// Save to file for persistence
fs.writeFileSync('knowledge-base.json', JSON.stringify(kb, null, 2));
```

### System Statistics

```typescript
const stats = neuroSymbolic.getStats();
console.log(`Processed ${stats.processedCount} inputs`);
console.log(`Average synergy: ${stats.averageSynergy.toFixed(3)}`);
console.log(`Knowledge base size: ${stats.knowledgeBaseSize}`);
```

## References

This implementation is inspired by:

- **OpenCog**: Cognitive architecture combining symbolic and neural AI
- **Reservoir Computing**: Echo-state networks and liquid state machines
- **Probabilistic Logic Networks**: Truth values and uncertain reasoning
- **Agent-Zero**: Tensor-based reasoning architectures

## Future Enhancements

Potential areas for extension:

1. Attention mechanisms for selective perception
2. More sophisticated inference rules (PLN formulas)
3. Online learning for dynamic knowledge updates
4. Multi-modal perception (vision, audio)
5. Distributed knowledge bases
6. Cognitive cycles (perception → reasoning → action)

## License

MIT License - See LICENSE file for details
