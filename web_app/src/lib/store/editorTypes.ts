export type EditorState = {
  baseBrushSize: number
  brushSizeScale: number
  renders: import("../types").StoredImage[]
  lineGroups: import("../types").LineGroup[]
  lastLineGroup: import("../types").LineGroup
  curLineGroup: import("../types").LineGroup
  extraMasks: import("../types").StoredImage[]
  prevExtraMasks: import("../types").StoredImage[]
  temporaryMasks: import("../types").StoredImage[]
  redoRenders: import("../types").StoredImage[]
  redoCurLines: import("../types").Line[]
  redoLineGroups: import("../types").LineGroup[]
}

export const editorDefaultValues: EditorState = {
  baseBrushSize: 12,
  brushSizeScale: 1,
  renders: [],
  lineGroups: [],
  lastLineGroup: [],
  curLineGroup: [],
  extraMasks: [],
  prevExtraMasks: [],
  temporaryMasks: [],
  redoRenders: [],
  redoCurLines: [],
  redoLineGroups: [],
}
