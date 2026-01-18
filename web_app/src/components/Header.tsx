import { PlayIcon } from "@radix-ui/react-icons"
import { useState } from "react"
import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"
import { useImage } from "@/hooks/useImage"

import { Popover, PopoverContent, PopoverTrigger } from "./ui/popover"
import PromptInput from "./PromptInput"
import { RotateCw, Image, Upload, Sparkles, Sun, Moon, Monitor } from "lucide-react"
import FileManager, { MASK_TAB } from "./FileManager"
import { getMediaBlob, getMediaFile } from "@/lib/api"
import { useStore } from "@/lib/states"
import SettingsDialog from "./Settings"
import { cn, fileToImage } from "@/lib/utils"
import { useToast } from "./ui/use-toast"
import { Switch } from "./ui/switch"
import { Label } from "./ui/label"
import { useTheme } from "next-themes"

const Header = () => {
  const [
    file,
    customMask,
    isInpainting,
    serverConfig,
    runMannually,
    enableUploadMask,
    model,
    setFile,
    setCustomFile,
    runInpainting,
    showPrevMask,
    hidePrevMask,
    imageHeight,
    imageWidth,
    handleFileManagerMaskSelect,
    // OpenAI mode (Epic 4)
    isOpenAIMode,
    setOpenAIMode,
    isOpenAIGenerating,
  ] = useStore((state) => [
    state.file,
    state.customMask,
    state.isInpainting,
    state.serverConfig,
    state.runMannually(),
    state.settings.enableUploadMask,
    state.settings.model,
    state.setFile,
    state.setCustomFile,
    state.runInpainting,
    state.showPrevMask,
    state.hidePrevMask,
    state.imageHeight,
    state.imageWidth,
    state.handleFileManagerMaskSelect,
    // OpenAI mode (Epic 4)
    state.openAIState.isOpenAIMode,
    state.setOpenAIMode,
    state.openAIState.isOpenAIGenerating,
  ])

  const { toast } = useToast()
  const [maskImage, maskImageLoaded] = useImage(customMask)
  const [openMaskPopover, setOpenMaskPopover] = useState(false)
  const { theme, setTheme } = useTheme()

  const handleRerunLastMask = () => {
    runInpainting()
  }

  const onRerunMouseEnter = () => {
    showPrevMask()
  }

  const onRerunMouseLeave = () => {
    hidePrevMask()
  }

  const handleOnPhotoClick = async (tab: string, filename: string) => {
    try {
      if (tab === MASK_TAB) {
        const maskBlob = await getMediaBlob(tab, filename)
        handleFileManagerMaskSelect(maskBlob)
      } else {
        const newFile = await getMediaFile(tab, filename)
        setFile(newFile)
      }
    } catch (e: any) {
      toast({
        variant: "destructive",
        description: e.message ? e.message : e.toString(),
      })
      return
    }
  }

  return (
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 border-b backdrop-filter backdrop-blur-md bg-background/70">
      <div className="flex items-center gap-1">
        {serverConfig.enableFileManager ? (
          <FileManager photoWidth={512} onPhotoClick={handleOnPhotoClick} />
        ) : (
          <></>
        )}

        <ImageUploadButton
          disabled={isInpainting}
          tooltip="Upload image"
          onFileUpload={(file) => {
            setFile(file)
          }}
        >
          <Image />
        </ImageUploadButton>

        <div
          className={cn([
            "flex items-center gap-1",
            file && enableUploadMask ? "visible" : "hidden",
          ])}
        >
          <ImageUploadButton
            disabled={isInpainting}
            tooltip="Upload custom mask"
            onFileUpload={async (file) => {
              let newCustomMask: HTMLImageElement | null = null
              try {
                newCustomMask = await fileToImage(file)
              } catch (e: any) {
                toast({
                  variant: "destructive",
                  description: e.message ? e.message : e.toString(),
                })
                return
              }
              if (
                newCustomMask.naturalHeight !== imageHeight ||
                newCustomMask.naturalWidth !== imageWidth
              ) {
                toast({
                  variant: "destructive",
                  description: `The size of the mask must same as image: ${imageWidth}x${imageHeight}`,
                })
                return
              }

              setCustomFile(file)
              if (!runMannually) {
                runInpainting()
              }
            }}
          >
            <Upload />
          </ImageUploadButton>

          {customMask ? (
            <Popover open={openMaskPopover}>
              <PopoverTrigger
                className="btn-primary side-panel-trigger"
                onMouseEnter={() => setOpenMaskPopover(true)}
                onMouseLeave={() => setOpenMaskPopover(false)}
                style={{
                  visibility: customMask ? "visible" : "hidden",
                  outline: "none",
                }}
              >
                <IconButton tooltip="Run custom mask">
                  <PlayIcon />
                </IconButton>
              </PopoverTrigger>
              <PopoverContent>
                {maskImageLoaded ? (
                  <img src={maskImage.src} alt="Custom mask" />
                ) : (
                  <></>
                )}
              </PopoverContent>
            </Popover>
          ) : (
            <></>
          )}
        </div>

        {file && !model.need_prompt ? (
          <IconButton
            disabled={isInpainting}
            tooltip="Rerun previous mask"
            onClick={handleRerunLastMask}
            onMouseEnter={onRerunMouseEnter}
            onMouseLeave={onRerunMouseLeave}
          >
            <RotateCw />
          </IconButton>
        ) : (
          <></>
        )}

        {/* Local/Cloud Mode Toggle */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-muted/50 border ml-2">
          <Label
            htmlFor="openai-mode"
            className={cn(
              "text-xs font-medium cursor-pointer transition-colors",
              !isOpenAIMode ? "text-foreground" : "text-muted-foreground"
            )}
          >
            Local
          </Label>
          <Switch
            id="openai-mode"
            checked={isOpenAIMode}
            onCheckedChange={setOpenAIMode}
            disabled={isInpainting || isOpenAIGenerating}
            className="data-[state=checked]:bg-primary"
          />
          <Label
            htmlFor="openai-mode"
            className={cn(
              "text-xs font-medium cursor-pointer transition-colors flex items-center gap-1",
              isOpenAIMode ? "text-foreground" : "text-muted-foreground"
            )}
          >
            <Sparkles className="h-3 w-3" />
            Cloud
          </Label>
        </div>
      </div>

      {/* Centered Prompt Input */}
      <div className="flex-1 flex justify-center">
        {!isOpenAIMode && model.need_prompt ? <PromptInput /> : <></>}
      </div>

      <div className="flex gap-1">
        <Shortcuts />
        {serverConfig.disableModelSwitch ? <></> : <SettingsDialog />}

        <IconButton
          tooltip={`Switch theme (${theme})`}
          onClick={() => {
            if (theme === "light") setTheme("dark");
            else if (theme === "dark") setTheme("system");
            else setTheme("light");
          }}
        >
          {theme === "light" && <Sun className="h-4 w-4" />}
          {theme === "dark" && <Moon className="h-4 w-4" />}
          {theme === "system" && <Monitor className="h-4 w-4" />}
        </IconButton>
      </div>
    </header>
  )
}

export default Header
