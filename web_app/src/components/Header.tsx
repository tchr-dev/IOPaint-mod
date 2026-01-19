import { IconButton, ImageUploadButton } from "@/components/ui/button"
import { Link } from "react-router-dom"
import Shortcuts from "@/components/Shortcuts"
import { Image, Sun, Moon, Monitor, Wrench } from "lucide-react"
import { useStore } from "@/lib/states"
import SettingsDialog from "./Settings"
import { useTheme } from "next-themes"

const Header = () => {
  const [
    isInpainting,
    setFile,
  ] = useStore((state) => [
    state.isInpainting,
    state.setFile,
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


      </div>

      <div className="flex gap-1">
        <Shortcuts />
        <Link to="/settings">
          <IconButton tooltip="Server Settings">
            <Wrench className="h-4 w-4" />
          </IconButton>
        </Link>
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
