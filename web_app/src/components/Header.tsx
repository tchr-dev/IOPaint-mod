import { IconButton, ImageUploadButton } from "@/components/ui/button"
import Shortcuts from "@/components/Shortcuts"
import { Image, Sparkles, Sun, Moon, Monitor } from "lucide-react"
import { useStore } from "@/lib/states"
import SettingsDialog from "./Settings"
import { cn } from "@/lib/utils"
import { Switch } from "./ui/switch"
import { Label } from "./ui/label"
import { useTheme } from "next-themes"

const Header = () => {
  const [
    isInpainting,
    setFile,
    // OpenAI mode (Epic 4)
    isOpenAIMode,
    setOpenAIMode,
    isOpenAIGenerating,
  ] = useStore((state) => [
    state.isInpainting,
    state.setFile,
    // OpenAI mode (Epic 4)
    state.openAIState.isOpenAIMode,
    state.setOpenAIMode,
    state.openAIState.isOpenAIGenerating,
  ])
  const { theme, setTheme } = useTheme()

  return (
    <header className="h-[60px] px-6 py-4 absolute top-[0] flex justify-between items-center w-full z-20 border-b backdrop-filter backdrop-blur-md bg-background/70">
      <div className="flex items-center gap-1">
        <ImageUploadButton
          disabled={isInpainting}
          tooltip="Upload image"
          onFileUpload={(file) => {
            setFile(file)
          }}
        >
          <Image />
        </ImageUploadButton>

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

      <div className="flex gap-1">
        <Shortcuts />
        <SettingsDialog />

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
