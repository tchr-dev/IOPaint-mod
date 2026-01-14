/**
 * OpenAI Components Index
 *
 * Exports all OpenAI-related components for the "Refine → Generate/Edit" workflow.
 *
 * Main Components:
 * - OpenAIGeneratePanel: Complete generation workflow panel
 * - OpenAIEditPanel: Edit flow with mask support (E4.2)
 *
 * Sub-Components:
 * - IntentInput: Raw intent/idea input
 * - PromptEditor: Refined prompt + negative prompt
 * - GenerationPresets: Draft/Final/Custom presets
 * - CostDisplay: Cost estimate and budget status
 * - CostWarningModal: High-cost confirmation
 */

export { IntentInput } from "./IntentInput"
export { PromptEditor } from "./PromptEditor"
export { GenerationPresets } from "./GenerationPresets"
export { CostDisplay } from "./CostDisplay"
export { CostWarningModal } from "./CostWarningModal"
export { OpenAIGeneratePanel } from "./OpenAIGeneratePanel"

// Phase 3: History/Gallery
export { GenerationHistory } from "./GenerationHistory"
export { HistoryItem } from "./HistoryItem"

// Phase 4: Edit flow
export { OpenAIEditPanel } from "./OpenAIEditPanel"
