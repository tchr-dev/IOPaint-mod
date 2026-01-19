import { beforeEach, describe, expect, it } from "vitest"

import { useStore } from "../../lib/states"
import type { Line, Point } from "../../lib/types"

const resetEditorState = () => {
  useStore.setState({
    file: null,
    paintByExampleFile: null,
    customMask: null,
    imageHeight: 0,
    imageWidth: 0,
    isInpainting: false,
    rendersCountBeforeInpaint: 0,
    isAdjustingMask: false,
    disableShortCuts: false,
    editorState: {
      baseBrushSize: 12,
      brushSizeScale: 1,
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
    },
    interactiveSegState: {
      isInteractiveSeg: false,
      tmpInteractiveSegMask: null,
      clicks: [],
    },
    cropperState: { x: 0, y: 0, width: 512, height: 512 },
    extenderState: { x: 0, y: 0, width: 512, height: 512 },
    isCropperExtenderResizing: false,
    settings: {
      ...useStore.getState().settings,
      enableManualInpainting: true,
    },
  })
}

const makeLine = (x: number, y: number): Line => ({
  size: 10,
  pts: [{ x, y }],
})

describe("Editor", () => {
  beforeEach(() => {
    localStorage.clear()
    resetEditorState()
  })

  describe("initial state", () => {
    it("calculates brush size correctly", () => {
      expect(useStore.getState().getBrushSize()).toBe(12)

      useStore.getState().setBaseBrushSize(24)
      expect(useStore.getState().getBrushSize()).toBe(24)
    })

    it("brush size scales with image size", () => {
      useStore.getState().setImageSize(1024, 1024)
      expect(useStore.getState().editorState.brushSizeScale).toBe(2)

      useStore.getState().setImageSize(512, 512)
      expect(useStore.getState().editorState.brushSizeScale).toBe(1)
    })
  })

  describe("brush size controls", () => {
    it("increases brush size", () => {
      useStore.getState().setBaseBrushSize(12)
      useStore.getState().increaseBaseBrushSize()
      expect(useStore.getState().editorState.baseBrushSize).toBe(22)
    })

    it("decreases brush size with larger steps for sizes above 10", () => {
      useStore.getState().setBaseBrushSize(50)
      useStore.getState().decreaseBaseBrushSize()
      expect(useStore.getState().editorState.baseBrushSize).toBe(40)
    })

    it("decreases brush size with smaller steps for sizes 1-10", () => {
      useStore.getState().setBaseBrushSize(5)
      useStore.getState().decreaseBaseBrushSize()
      expect(useStore.getState().editorState.baseBrushSize).toBe(2)
    })

    it("allows negative brush size when decreasing from small values", () => {
      useStore.getState().setBaseBrushSize(1)
      useStore.getState().decreaseBaseBrushSize()
      expect(useStore.getState().editorState.baseBrushSize).toBe(-2)
    })

    it("caps brush size at maximum (200)", () => {
      useStore.getState().setBaseBrushSize(200)
      useStore.getState().increaseBaseBrushSize()
      expect(useStore.getState().editorState.baseBrushSize).toBe(200)
    })
  })

  describe("canvas mouse events", () => {
    it("handles mouse down to start a new line group", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          enableManualInpainting: true,
        },
      })

      const point: Point = { x: 100, y: 100 }
      useStore.getState().handleCanvasMouseDown(point)

      expect(useStore.getState().editorState.curLineGroup).toHaveLength(1)
      expect(useStore.getState().editorState.curLineGroup[0].pts).toHaveLength(1)
      expect(useStore.getState().editorState.curLineGroup[0].pts[0]).toEqual(point)
    })

    it("handles mouse move to extend current line", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          enableManualInpainting: true,
        },
      })

      const point1: Point = { x: 100, y: 100 }
      const point2: Point = { x: 105, y: 105 }

      useStore.getState().handleCanvasMouseDown(point1)
      useStore.getState().handleCanvasMouseMove(point2)

      expect(useStore.getState().editorState.curLineGroup[0].pts).toHaveLength(2)
      expect(useStore.getState().editorState.curLineGroup[0].pts[1]).toEqual(point2)
    })

    it("handles mouse move when no active line group", () => {
      expect(() => {
        useStore.getState().handleCanvasMouseMove({ x: 100, y: 100 })
      }).not.toThrow()
    })
  })

  describe("mask operations", () => {
    it("clears mask and current line group", () => {
      useStore.setState({
        editorState: {
          ...useStore.getState().editorState,
          curLineGroup: [makeLine(1, 1), makeLine(2, 2)],
          extraMasks: [{ src: "data:image/png;base64,test", id: "1" } as any],
        },
      })

      useStore.getState().clearMask()

      expect(useStore.getState().editorState.curLineGroup).toHaveLength(0)
      expect(useStore.getState().editorState.extraMasks).toHaveLength(0)
    })
  })

  describe("undo/redo state", () => {
    it("resets redo state manually", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          enableManualInpainting: true,
        },
        editorState: {
          ...useStore.getState().editorState,
          redoRenders: [{ src: "data:image/png;base64,test", id: "1" } as any],
          redoCurLines: [makeLine(1, 1)],
          redoLineGroups: [[makeLine(2, 2)]],
        },
      })

      useStore.getState().resetRedoState()

      expect(useStore.getState().editorState.redoRenders).toHaveLength(0)
      expect(useStore.getState().editorState.redoCurLines).toHaveLength(0)
      expect(useStore.getState().editorState.redoLineGroups).toHaveLength(0)
    })

    it("undoDisabled returns true when no renders exist in auto mode", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          enableManualInpainting: false,
        },
      })
      expect(useStore.getState().undoDisabled()).toBe(true)
    })

    it("redoDisabled returns true when no redo renders exist", () => {
      expect(useStore.getState().redoDisabled()).toBe(true)
    })
  })

  describe("processing state", () => {
    it("returns true when inpainting", () => {
      useStore.setState({ isInpainting: true })
      expect(useStore.getState().getIsProcessing()).toBe(true)
    })

    it("returns true when adjusting mask", () => {
      useStore.setState({ isAdjustingMask: true })
      expect(useStore.getState().getIsProcessing()).toBe(true)
    })

    it("returns false when no processing is happening", () => {
      useStore.setState({ isInpainting: false, isAdjustingMask: false })
      expect(useStore.getState().getIsProcessing()).toBe(false)
    })
  })

  describe("model type detection", () => {
    it("returns false for inpaint model type", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          model: { ...useStore.getState().settings.model, model_type: "inpaint" },
        },
      })
      expect(useStore.getState().isSD()).toBe(false)
    })

    it("returns true for non-inpaint model types", () => {
      useStore.setState({
        settings: {
          ...useStore.getState().settings,
          model: { ...useStore.getState().settings.model, model_type: "diffusers_sd" },
        },
      })
      expect(useStore.getState().isSD()).toBe(true)
    })
  })

  describe("editor state updates", () => {
    it("updateEditorState merges updates correctly", () => {
      useStore.getState().updateEditorState({ baseBrushSize: 50 })
      expect(useStore.getState().editorState.baseBrushSize).toBe(50)
      expect(useStore.getState().editorState.brushSizeScale).toBe(1)
    })

    it("cleanCurLineGroup clears current line group", () => {
      useStore.setState({
        editorState: {
          ...useStore.getState().editorState,
          curLineGroup: [makeLine(1, 1), makeLine(2, 2)],
        },
      })

      useStore.getState().cleanCurLineGroup()

      expect(useStore.getState().editorState.curLineGroup).toHaveLength(0)
    })
  })

  describe("settings mutations", () => {
    it("updateSettings merges settings correctly", () => {
      useStore.getState().updateSettings({ prompt: "test prompt" })
      expect(useStore.getState().settings.prompt).toBe("test prompt")
      expect(useStore.getState().settings.seed).toBe(42)
    })

    it("setSeed updates seed setting", () => {
      useStore.getState().setSeed(123)
      expect(useStore.getState().settings.seed).toBe(123)
    })
  })

  describe("cropper state", () => {
    it("updates cropper x coordinate", () => {
      useStore.getState().setCropperX(100)
      expect(useStore.getState().cropperState.x).toBe(100)
    })

    it("updates cropper dimensions", () => {
      useStore.getState().setCropperWidth(256)
      useStore.getState().setCropperHeight(256)
      expect(useStore.getState().cropperState.width).toBe(256)
      expect(useStore.getState().cropperState.height).toBe(256)
    })
  })

  describe("extender state", () => {
    it("updates extender coordinates", () => {
      useStore.getState().setExtenderX(50)
      useStore.getState().setExtenderY(50)
      expect(useStore.getState().extenderState.x).toBe(50)
      expect(useStore.getState().extenderState.y).toBe(50)
    })

    it("updates extender dimensions", () => {
      useStore.getState().setExtenderWidth(1024)
      useStore.getState().setExtenderHeight(1024)
      expect(useStore.getState().extenderState.width).toBe(1024)
      expect(useStore.getState().extenderState.height).toBe(1024)
    })
  })

  describe("interactive seg state", () => {
    it("updates interactive seg state", () => {
      useStore.getState().updateInteractiveSegState({ isInteractiveSeg: true })
      expect(useStore.getState().interactiveSegState.isInteractiveSeg).toBe(true)
    })

    it("resets interactive seg state", () => {
      useStore.setState({
        interactiveSegState: {
          isInteractiveSeg: true,
          tmpInteractiveSegMask: { src: "test", id: "1" } as any,
          clicks: [[100, 100]],
        },
      })

      useStore.getState().resetInteractiveSegState()

      expect(useStore.getState().interactiveSegState.isInteractiveSeg).toBe(false)
      expect(useStore.getState().interactiveSegState.tmpInteractiveSegMask).toBeNull()
      expect(useStore.getState().interactiveSegState.clicks).toHaveLength(0)
    })
  })

  describe("file manager state", () => {
    it("updates file manager state", () => {
      useStore.getState().updateFileManagerState({ searchText: "test" })
      expect(useStore.getState().fileManagerState.searchText).toBe("test")
    })

    it("toggles layout between masonry and rows", () => {
      useStore.getState().updateFileManagerState({ layout: "masonry" })
      expect(useStore.getState().fileManagerState.layout).toBe("masonry")
    })
  })
})
