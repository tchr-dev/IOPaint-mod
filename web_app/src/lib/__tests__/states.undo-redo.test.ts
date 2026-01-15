import { beforeEach, describe, expect, it } from "vitest"

import { useStore } from "../states"
import type { Line, LineGroup } from "../types"

const makeLine = (x: number, y: number): Line => ({
  size: 10,
  pts: [{ x, y }],
})

const resetEditorState = (overrides?: Partial<ReturnType<typeof useStore.getState>["editorState"]>) => {
  const state = useStore.getState()
  useStore.setState({
    editorState: {
      ...state.editorState,
      renders: [],
      extraMasks: [],
      prevExtraMasks: [],
      temporaryMasks: [],
      lineGroups: [],
      lastLineGroup: [],
      curLineGroup: [],
      redoRenders: [],
      redoCurLines: [],
      redoLineGroups: [],
      ...overrides,
    },
  })
}

const setManualMode = (enabled: boolean) => {
  const state = useStore.getState()
  useStore.setState({
    settings: {
      ...state.settings,
      enableManualInpainting: enabled,
      model: {
        ...state.settings.model,
        model_type: "inpaint",
      },
    },
  })
}

beforeEach(() => {
  setManualMode(true)
  resetEditorState()
})

describe("editor undo/redo", () => {
  it("undo/redo strokes in manual mode", () => {
    const lineGroup: LineGroup = [makeLine(1, 1), makeLine(2, 2)]
    resetEditorState({ curLineGroup: [...lineGroup] })

    const state = useStore.getState()
    expect(state.undoDisabled()).toBe(false)

    state.undo()
    expect(useStore.getState().editorState.curLineGroup).toHaveLength(1)
    expect(useStore.getState().editorState.redoCurLines).toHaveLength(1)
    expect(useStore.getState().redoDisabled()).toBe(false)

    state.redo()
    expect(useStore.getState().editorState.curLineGroup).toHaveLength(2)
    expect(useStore.getState().editorState.redoCurLines).toHaveLength(0)
  })

  it("undo/redo renders in auto mode", () => {
    setManualMode(false)
    const imageA = new Image()
    const imageB = new Image()
    const lineGroupA: LineGroup = [makeLine(1, 1)]
    const lineGroupB: LineGroup = [makeLine(2, 2)]

    resetEditorState({
      renders: [imageA, imageB],
      lineGroups: [lineGroupA, lineGroupB],
    })

    const state = useStore.getState()
    expect(state.undoDisabled()).toBe(false)

    state.undo()
    expect(useStore.getState().editorState.renders).toHaveLength(1)
    expect(useStore.getState().editorState.lineGroups).toHaveLength(1)
    expect(useStore.getState().editorState.redoRenders).toHaveLength(1)
    expect(useStore.getState().editorState.redoLineGroups).toHaveLength(1)

    state.redo()
    expect(useStore.getState().editorState.renders).toHaveLength(2)
    expect(useStore.getState().editorState.lineGroups).toHaveLength(2)
    expect(useStore.getState().editorState.redoRenders).toHaveLength(0)
    expect(useStore.getState().editorState.redoLineGroups).toHaveLength(0)
  })

  it("undo disabled when no history exists", () => {
    resetEditorState()
    setManualMode(false)

    expect(useStore.getState().undoDisabled()).toBe(true)
    expect(useStore.getState().redoDisabled()).toBe(true)
  })
})
