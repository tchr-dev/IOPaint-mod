import { SyntheticEvent } from "react"
import {
  ReactZoomPanPinchContentRef,
  TransformComponent,
  TransformWrapper,
} from "react-zoom-pan-pinch"
import { cn } from "@/lib/utils"
import Cropper from "../Cropper"
import Extender from "../Extender"
import { InteractiveSegPoints } from "../InteractiveSeg"

interface CanvasViewProps {
  imageWidth: number
  imageHeight: number
  isProcessing: boolean
  sliderPos: number
  isChangingBrushSizeByWheel: boolean
  isPanning: boolean
  panned: boolean
  minScale: number
  settings: {
    showCropper: boolean
    showExtender: boolean
  }
  interactiveSegState: {
    isInteractiveSeg: boolean
  }
  getCursor: () => string | undefined
  getCurScale: () => number
  toggleShowBrush: (newState: boolean) => void
  onMouseDown: (ev: SyntheticEvent) => void
  onMouseUp: (ev: SyntheticEvent) => void
  onMouseDrag: (ev: SyntheticEvent) => void
  setPanned: (value: React.SetStateAction<boolean>) => void
  setScale: (value: React.SetStateAction<number>) => void
  viewportRef: React.MutableRefObject<ReactZoomPanPinchContentRef | null>
  initialCentered: boolean
}

const TRANSITION_DURATION_MS = 300

export function CanvasView({
  imageWidth,
  imageHeight,
  isProcessing,
  sliderPos,
  isChangingBrushSizeByWheel,
  isPanning,
  panned,
  minScale,
  settings,
  interactiveSegState,
  getCursor,
  getCurScale,
  toggleShowBrush,
  onMouseDown,
  onMouseUp,
  onMouseDrag,
  setPanned,
  setScale,
  viewportRef,
  initialCentered,
}: CanvasViewProps) {
  return (
    <TransformWrapper
      ref={(r) => {
        if (r) {
          viewportRef.current = r
        }
      }}
      panning={{ disabled: !isPanning, velocityDisabled: true }}
      wheel={{ step: 0.05, wheelDisabled: isChangingBrushSizeByWheel }}
      centerZoomedOut
      alignmentAnimation={{ disabled: true }}
      centerOnInit
      limitToBounds={false}
      doubleClick={{ disabled: true }}
      initialScale={minScale}
      minScale={minScale * 0.3}
      onPanning={() => {
        if (!panned) {
          setPanned(true)
        }
      }}
      onZoom={(ref) => {
        setScale(ref.state.scale)
      }}
    >
      <TransformComponent
        contentStyle={{
          visibility: initialCentered ? "visible" : "hidden",
        }}
      >
        <div className="grid [grid-template-areas:'editor-content'] gap-y-4">
          <canvas
            className="[grid-area:editor-content]"
            style={{
              clipPath: `inset(0 ${sliderPos}% 0 0)`,
              transition: `clip-path ${TRANSITION_DURATION_MS}ms`,
            }}
          />
          <canvas
            className={cn(
              "[grid-area:editor-content]",
              isProcessing ? "pointer-events-none animate-pulse duration-600" : ""
            )}
            style={{
              cursor: getCursor(),
              clipPath: `inset(0 ${sliderPos}% 0 0)`,
              transition: `clip-path ${TRANSITION_DURATION_MS}ms`,
            }}
            onContextMenu={(e) => {
              e.preventDefault()
            }}
            onMouseOver={() => {
              toggleShowBrush(true)
            }}
            onFocus={() => toggleShowBrush(true)}
            onMouseLeave={() => toggleShowBrush(false)}
            onMouseDown={onMouseDown}
            onMouseUp={onMouseUp}
            onMouseMove={onMouseDrag}
            onTouchStart={onMouseDown}
            onTouchEnd={onMouseUp}
            onTouchMove={onMouseDrag}
          />
        </div>

        <Cropper
          maxHeight={imageHeight}
          maxWidth={imageWidth}
          minHeight={Math.min(512, imageHeight)}
          minWidth={Math.min(512, imageWidth)}
          scale={getCurScale()}
          show={settings.showCropper}
        />

        <Extender
          minHeight={Math.min(512, imageHeight)}
          minWidth={Math.min(512, imageWidth)}
          scale={getCurScale()}
          show={settings.showExtender}
        />

        {interactiveSegState.isInteractiveSeg ? (
          <InteractiveSegPoints />
        ) : null}
      </TransformComponent>
    </TransformWrapper>
  )
}
