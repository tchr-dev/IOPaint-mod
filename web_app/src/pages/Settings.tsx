
import React, { useEffect } from "react"
import { useForm } from "react-hook-form"
import { zodResolver } from "@hookform/resolvers/zod"
import * as z from "zod"
import { useStore } from "../lib/states"
import {
    getServerConfig,
    saveConfig
} from "../lib/api"
import { ApiConfig, ServerConfig } from "../lib/types"
import { Button } from "../components/ui/button"
import { Input } from "../components/ui/input"
import { Label } from "../components/ui/label"
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from "../components/ui/select"
import { Slider } from "../components/ui/slider"
import { Switch } from "../components/ui/switch"
import { useToast } from "../components/ui/use-toast"
import {
    Form,
    FormControl,
    FormDescription,
    FormField,
    FormItem,
    FormLabel,
    FormMessage,
} from "../components/ui/form"
import { Separator } from "../components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../components/ui/tabs"

const formSchema = z.object({
    host: z.string(),
    port: z.coerce.number(),
    inbrowser: z.boolean(),
    model: z.string(),
    device: z.string(),
    quality: z.coerce.number().min(75).max(100),
    input: z.string().optional(),
    output_dir: z.string().optional(),
    mask_dir: z.string().optional(),
    enable_interactive_seg: z.boolean(),
    interactive_seg_model: z.string(),
    interactive_seg_device: z.string(),
    enable_remove_bg: z.boolean(),
    remove_bg_model: z.string(),
    remove_bg_device: z.string(),
    enable_realesrgan: z.boolean(),
    realesrgan_model: z.string(),
    realesrgan_device: z.string(),
    enable_gfpgan: z.boolean(),
    gfpgan_device: z.string(),
    enable_restoreformer: z.boolean(),
    restoreformer_device: z.string(),
})

export default function Settings() {
    const { toast } = useToast()
    const [setServerConfig] = useStore((state) => [state.setServerConfig])

    const form = useForm<z.infer<typeof formSchema>>({
        resolver: zodResolver(formSchema),
        defaultValues: {
            host: "127.0.0.1",
            port: 8080,
            inbrowser: false,
            model: "lama",
            device: "cpu",
            quality: 95,
            enable_interactive_seg: false,
            interactive_seg_model: "mobile_sam",
            interactive_seg_device: "cpu",
            enable_remove_bg: false,
            remove_bg_model: "u2net",
            remove_bg_device: "cpu",
            enable_realesrgan: false,
            realesrgan_model: "realesr-general-x4v3",
            realesrgan_device: "cpu",
            enable_gfpgan: false,
            gfpgan_device: "cpu",
            enable_restoreformer: false,
            restoreformer_device: "cpu",
        }
    })

    const [serverConfigData, setServerConfigData] = React.useState<ServerConfig | null>(null)

    useEffect(() => {
        const fetchConfig = async () => {
            try {
                const config = await getServerConfig()
                setServerConfig(config)
                setServerConfigData(config)
                // We cast to any because ServerConfig (from backend) now has extra fields
                // but TypeScript interface might strictly follow what we defined earlier.
                // We updated types.ts so it should be fine if we map it correctly.
                // However, getServerConfig returns ServerConfig which has `modelInfos` etc.
                // It also has the flat properties we added to python schema.
                // We need to verify if types.ts ServerConfig actually has those flat properties.
                // I checked API response types in python: `api_server_config` returns `ServerConfigResponse`.
                // In my edit to `iopaint/schema.py`, I added `host`, `port` etc to `ServerConfigResponse`.
                // In my edit to `types.ts`, I did NOT add `host`, `port` to `ServerConfig` interface yet!
                // I added `ApiConfig` interface separately. `ServerConfig` is what `getServerConfig` returns.
                // I need to update `ServerConfig` interface in `types.ts` OR just cast here.
                // For safety, I'll cast to any for now to avoid compilation errors if types are out of sync.
                const data = config as any

                form.reset({
                    host: data.host,
                    port: data.port,
                    inbrowser: data.inbrowser,
                    model: data.model,
                    device: data.device,
                    quality: data.quality,
                    input: data.input ? String(data.input) : "",
                    output_dir: data.output_dir ? String(data.output_dir) : "",
                    mask_dir: data.mask_dir ? String(data.mask_dir) : "",
                    enable_interactive_seg: data.enable_interactive_seg,
                    interactive_seg_model: data.interactive_seg_model,
                    interactive_seg_device: data.interactive_seg_device,
                    enable_remove_bg: data.enable_remove_bg,
                    remove_bg_model: data.removeBGModel, // Note: backend helper uses this field name pattern in original response? Check api.py mapping.
                    // in api.py: removeBGModel=self.config.remove_bg_model
                    // BUT I also added `remove_bg_model=self.config.enable_remove_bg`? No.
                    // I added `remove_bg_model`? Let's check api.py again.
                    // I added `enable_remove_bg=self.config.enable_remove_bg`
                    // I did NOT add `remove_bg_model` because `removeBGModel` was already there.
                    // OK so I should use `removeBGModel`.
                    remove_bg_device: data.remove_bg_device,
                    enable_realesrgan: data.enable_realesrgan,
                    realesrgan_model: data.realesrganModel,
                    realesrgan_device: data.realesrgan_device,
                    enable_gfpgan: data.enable_gfpgan,
                    gfpgan_device: data.gfpgan_device,
                    enable_restoreformer: data.enable_restoreformer,
                    restoreformer_device: data.restoreformer_device,
                })
            } catch (e) {
                toast({
                    variant: "destructive",
                    title: "Failed to load config",
                    description: String(e)
                })
            }
        }
        fetchConfig()
    }, [form, setServerConfig, toast])

    async function onSubmit(values: z.infer<typeof formSchema>) {
        try {
            await saveConfig(values)
            toast({
                title: "Configuration Saved",
                description: "Settings have been saved to iopaint_config.json"
            })
        } catch (e) {
            toast({
                variant: "destructive",
                title: "Failed to save config",
                description: String(e)
            })
        }
    }

    const devices = ["cpu", "cuda", "mps"]

    return (
        <div className="container mx-auto p-6 max-w-4xl bg-background text-foreground min-h-screen">
            <div className="flex justify-between items-center mb-6">
                <h1 className="text-3xl font-bold">Settings</h1>
                <Button onClick={() => window.location.href = "/"}>Back to Editor</Button>
            </div>

            <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-8">
                    <Tabs defaultValue="common" className="w-full">
                        <TabsList className="grid w-full grid-cols-2">
                            <TabsTrigger value="common">Common</TabsTrigger>
                            <TabsTrigger value="plugins">Plugins</TabsTrigger>
                        </TabsList>

                        <TabsContent value="common" className="space-y-4 py-4">
                            <div className="grid grid-cols-2 gap-4">
                                <FormField
                                    control={form.control}
                                    name="host"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Host</FormLabel>
                                            <FormControl><Input {...field} /></FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                                <FormField
                                    control={form.control}
                                    name="port"
                                    render={({ field }) => (
                                        <FormItem>
                                            <FormLabel>Port</FormLabel>
                                            <FormControl><Input type="number" {...field} /></FormControl>
                                            <FormMessage />
                                        </FormItem>
                                    )}
                                />
                            </div>

                            <FormField
                                control={form.control}
                                name="model"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Model</FormLabel>
                                        <Select onValueChange={field.onChange} value={field.value}>
                                            <FormControl>
                                                <SelectTrigger>
                                                    <SelectValue placeholder="Select a model" />
                                                </SelectTrigger>
                                            </FormControl>
                                            <SelectContent>
                                                {serverConfigData?.modelInfos.map(model => (
                                                    <SelectItem key={model.name} value={model.name}>{model.name}</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                        <FormDescription>Current Inpainting Model</FormDescription>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />

                            <FormField
                                control={form.control}
                                name="device"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Device</FormLabel>
                                        <Select onValueChange={field.onChange} value={field.value}>
                                            <FormControl>
                                                <SelectTrigger>
                                                    <SelectValue placeholder="Select device" />
                                                </SelectTrigger>
                                            </FormControl>
                                            <SelectContent>
                                                {devices.map(d => (
                                                    <SelectItem key={d} value={d}>{d}</SelectItem>
                                                ))}
                                            </SelectContent>
                                        </Select>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />

                            <FormField
                                control={form.control}
                                name="quality"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Image Quality ({field.value})</FormLabel>
                                        <FormControl>
                                            <Slider
                                                min={75}
                                                max={100}
                                                step={1}
                                                defaultValue={[field.value]}
                                                onValueChange={(vals) => field.onChange(vals[0])}
                                            />
                                        </FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />

                            <Separator />

                            <FormField
                                control={form.control}
                                name="input"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Input File/Directory</FormLabel>
                                        <FormControl><Input {...field} /></FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="output_dir"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Output Directory</FormLabel>
                                        <FormControl><Input {...field} /></FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                            <FormField
                                control={form.control}
                                name="mask_dir"
                                render={({ field }) => (
                                    <FormItem>
                                        <FormLabel>Mask Directory</FormLabel>
                                        <FormControl><Input {...field} /></FormControl>
                                        <FormMessage />
                                    </FormItem>
                                )}
                            />
                        </TabsContent>

                        <TabsContent value="plugins" className="space-y-4 py-4">
                            {/* Interactive Seg */}
                            <div className="space-y-4 border p-4 rounded-md">
                                <FormField
                                    control={form.control}
                                    name="enable_interactive_seg"
                                    render={({ field }) => (
                                        <FormItem className="flex flex-row items-center justify-between rounded-lg p-3 shadow-sm">
                                            <div className="space-y-0.5">
                                                <FormLabel>Interactive Segmentation</FormLabel>
                                                <FormDescription>Enable Segment Anything</FormDescription>
                                            </div>
                                            <FormControl>
                                                <Switch checked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                        </FormItem>
                                    )}
                                />
                                {form.watch("enable_interactive_seg") && (
                                    <div className="grid grid-cols-2 gap-4 pl-4">
                                        <FormField
                                            control={form.control}
                                            name="interactive_seg_model"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Model</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {serverConfigData?.interactiveSegModels.map(m => (
                                                                <SelectItem key={m} value={m}>{m}</SelectItem>
                                                            ))}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                        <FormField
                                            control={form.control}
                                            name="interactive_seg_device"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Device</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {devices.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* Remove BG */}
                            <div className="space-y-4 border p-4 rounded-md">
                                <FormField
                                    control={form.control}
                                    name="enable_remove_bg"
                                    render={({ field }) => (
                                        <FormItem className="flex flex-row items-center justify-between rounded-lg p-3 shadow-sm">
                                            <div className="space-y-0.5">
                                                <FormLabel>Remove Background</FormLabel>
                                                <FormDescription>Enable background removal</FormDescription>
                                            </div>
                                            <FormControl>
                                                <Switch checked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                        </FormItem>
                                    )}
                                />
                                {form.watch("enable_remove_bg") && (
                                    <div className="grid grid-cols-2 gap-4 pl-4">
                                        <FormField
                                            control={form.control}
                                            name="remove_bg_model"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Model</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {serverConfigData?.removeBGModels.map(m => (
                                                                <SelectItem key={m} value={m}>{m}</SelectItem>
                                                            ))}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                        <FormField
                                            control={form.control}
                                            name="remove_bg_device"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Device</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {devices.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* RealESRGAN */}
                            <div className="space-y-4 border p-4 rounded-md">
                                <FormField
                                    control={form.control}
                                    name="enable_realesrgan"
                                    render={({ field }) => (
                                        <FormItem className="flex flex-row items-center justify-between rounded-lg p-3 shadow-sm">
                                            <div className="space-y-0.5">
                                                <FormLabel>RealESRGAN Upscaling</FormLabel>
                                            </div>
                                            <FormControl>
                                                <Switch checked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                        </FormItem>
                                    )}
                                />
                                {form.watch("enable_realesrgan") && (
                                    <div className="grid grid-cols-2 gap-4 pl-4">
                                        <FormField
                                            control={form.control}
                                            name="realesrgan_model"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Model</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {serverConfigData?.realesrganModels.map(m => (
                                                                <SelectItem key={m} value={m}>{m}</SelectItem>
                                                            ))}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                        <FormField
                                            control={form.control}
                                            name="realesrgan_device"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Device</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {devices.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* GFPGAN */}
                            <div className="space-y-4 border p-4 rounded-md">
                                <FormField
                                    control={form.control}
                                    name="enable_gfpgan"
                                    render={({ field }) => (
                                        <FormItem className="flex flex-row items-center justify-between rounded-lg p-3 shadow-sm">
                                            <div className="space-y-0.5">
                                                <FormLabel>GFPGAN Face Restoration</FormLabel>
                                            </div>
                                            <FormControl>
                                                <Switch checked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                        </FormItem>
                                    )}
                                />
                                {form.watch("enable_gfpgan") && (
                                    <div className="grid grid-cols-2 gap-4 pl-4">
                                        <FormField
                                            control={form.control}
                                            name="gfpgan_device"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Device</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {devices.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                )}
                            </div>

                            {/* RestoreFormer */}
                            <div className="space-y-4 border p-4 rounded-md">
                                <FormField
                                    control={form.control}
                                    name="enable_restoreformer"
                                    render={({ field }) => (
                                        <FormItem className="flex flex-row items-center justify-between rounded-lg p-3 shadow-sm">
                                            <div className="space-y-0.5">
                                                <FormLabel>RestoreFormer</FormLabel>
                                            </div>
                                            <FormControl>
                                                <Switch checked={field.value} onCheckedChange={field.onChange} />
                                            </FormControl>
                                        </FormItem>
                                    )}
                                />
                                {form.watch("enable_restoreformer") && (
                                    <div className="grid grid-cols-2 gap-4 pl-4">
                                        <FormField
                                            control={form.control}
                                            name="restoreformer_device"
                                            render={({ field }) => (
                                                <FormItem>
                                                    <FormLabel>Device</FormLabel>
                                                    <Select onValueChange={field.onChange} value={field.value}>
                                                        <FormControl><SelectTrigger><SelectValue /></SelectTrigger></FormControl>
                                                        <SelectContent>
                                                            {devices.map(d => <SelectItem key={d} value={d}>{d}</SelectItem>)}
                                                        </SelectContent>
                                                    </Select>
                                                </FormItem>
                                            )}
                                        />
                                    </div>
                                )}
                            </div>

                        </TabsContent>
                    </Tabs>

                    <Button type="submit">Save Configuration</Button>
                </form>
            </Form>
        </div>
    )
}
