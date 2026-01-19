import { useToggle } from "react-use"
import { useStore } from "@/lib/states"
import { Separator } from "../ui/separator"
import { ScrollArea } from "../ui/scroll-area"
import { Sheet, SheetContent, SheetHeader, SheetTrigger } from "../ui/sheet"
import { ChevronLeft, ChevronRight } from "lucide-react"
import { Button } from "../ui/button"
import useHotKey from "@/hooks/useHotkey"
import { RowContainer } from "./LabelTitle"
import SimplifiedOptions from "./SimplifiedOptions"

const SidePanel = () => {
  const [windowSize] = useStore((state) => [
    state.windowSize,
  ])

  const [open, toggleOpen] = useToggle(true)

  useHotKey("c", () => {
    toggleOpen()
  })

  const renderSidePanelOptions = () => {
    return <SimplifiedOptions />
  }

  const getPanelTitle = () => {
    return "Quality Presets"
  }

  return (
    <Sheet open={open} modal={false}>
      <SheetTrigger
        tabIndex={-1}
        className="z-10 outline-none absolute top-[68px] right-6 rounded-lg border bg-background"
        hidden={open}
      >
        <Button
          variant="ghost"
          size="icon"
          asChild
          className="p-1.5"
          onClick={toggleOpen}
        >
          <ChevronLeft strokeWidth={1} />
        </Button>
      </SheetTrigger>
        <SheetContent
          side="right"
          className="min-w-[286px] max-w-full mt-[60px] outline-none px-3"
          onOpenAutoFocus={(event) => event.preventDefault()}
          onPointerDownOutside={(event) => event.preventDefault()}
        >
        <SheetHeader>
          <RowContainer>
            <div className="overflow-hidden mr-8">
              {getPanelTitle()}
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="border h-6 w-6"
              onClick={toggleOpen}
            >
              <ChevronRight strokeWidth={1} />
            </Button>
          </RowContainer>
          <Separator />
        </SheetHeader>
        <ScrollArea style={{ height: windowSize.height - 160 }}>
          {renderSidePanelOptions()}
        </ScrollArea>
      </SheetContent>
    </Sheet>
  )
}

export default SidePanel
