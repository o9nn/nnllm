/**
 * Neuro-Symbolic AI Module
 *
 * OpenCog-inspired neuro-symbolic architecture combining:
 * - Echo-State Neural Network (ESN) for perception
 * - Tensor-Logic Reasoning Engine for symbolic reasoning
 * - Cognitive synergy integration
 */

export { EchoStateNetwork, ESNConfig } from './echo-state-network.js';
export {
  TensorLogicEngine,
  AtomSpace,
  Atom,
  TruthValue,
  Link,
  LogicOperator,
  InferenceRule,
} from './tensor-logic-engine.js';
export {
  NeuroSymbolicLLM,
  NeuroSymbolicConfig,
  ProcessingResult,
} from './neuro-symbolic-llm.js';
