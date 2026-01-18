import { SortBy, SortOrder } from "../types"

export type FileManagerState = {
  sortBy: SortBy
  sortOrder: SortOrder
  layout: "rows" | "masonry"
  searchText: string
  inputDirectory: string
  outputDirectory: string
}

export type CropperState = {
  x: number
  y: number
  width: number
  height: number
}

export type InteractiveSegState = {
  isInteractiveSeg: boolean
  tmpInteractiveSegMask: import("../types").StoredImage | null
  clicks: number[][]
}

export const fileManagerDefaultValues: FileManagerState = {
  sortBy: SortBy.CTIME,
  sortOrder: SortOrder.DESCENDING,
  layout: "masonry",
  searchText: "",
  inputDirectory: "",
  outputDirectory: "",
}

export const cropperDefaultValues: CropperState = {
  x: 0,
  y: 0,
  width: 512,
  height: 512,
}

export const interactiveSegDefaultValues: InteractiveSegState = {
  isInteractiveSeg: false,
  tmpInteractiveSegMask: null,
  clicks: [],
}
