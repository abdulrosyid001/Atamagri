"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarInset,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Thermometer,
  Droplets,
  Wind,
  Sun,
  Zap,
  Gauge,
  CloudRain,
  BarChart3,
  Settings,
  User,
  Download,
  Leaf,
  ChevronDown,
  Bell,
  RefreshCw,
  MapPin,
  Newspaper,
  Eye,
  EyeOff,
<<<<<<< HEAD
  Airplay,
  LucideIcon
} from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import Link from "next/link"
import Image from "next/image"
=======
} from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts"
import Link from "next/link"
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
import { useState } from "react"

// Mock data for charts
const temperatureData = [
  { time: "00:00", value: 24.5 },
  { time: "04:00", value: 22.8 },
  { time: "08:00", value: 28.3 },
  { time: "12:00", value: 35.1 },
  { time: "16:00", value: 33.7 },
  { time: "20:00", value: 29.2 },
]

const humidityData = [
  { time: "00:00", value: 78.2 },
  { time: "04:00", value: 82.1 },
  { time: "08:00", value: 65.4 },
  { time: "12:00", value: 57.7 },
  { time: "16:00", value: 61.3 },
  { time: "20:00", value: 69.8 },
]

// Weather forecast data
const weatherForecast = [
  { time: "12:00", temp: 35, wind: 8, rain: 10, icon: "☀️" },
  { time: "15:00", temp: 33, wind: 12, rain: 15, icon: "⛅" },
  { time: "18:00", temp: 29, wind: 6, rain: 5, icon: "☀️" },
  { time: "21:00", temp: 26, wind: 4, rain: 0, icon: "🌙" },
  { time: "00:00", temp: 24, wind: 3, rain: 0, icon: "🌙" },
  { time: "03:00", temp: 22, wind: 5, rain: 20, icon: "🌧️" },
]

// News articles data
const newsArticles = [
  {
    id: 1,
    title: "Optimizing Rice Irrigation in Dry Season",
    category: "Tanaman",
    excerpt: "Learn effective water management techniques for rice cultivation during drought periods.",
<<<<<<< HEAD
    image: "/drought.png",
=======
    image: "/placeholder.svg?height=100&width=150",
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    readTime: "5 min read",
  },
  {
    id: 2,
    title: "Weather Patterns and Crop Planning",
    category: "Cuaca",
    excerpt: "Understanding seasonal weather changes for better agricultural planning and decision making.",
<<<<<<< HEAD
    image: "/weather.png",
=======
    image: "/placeholder.svg?height=100&width=150",
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    readTime: "7 min read",
  },
  {
    id: 3,
    title: "Soil Health Management Techniques",
    category: "Tanah",
    excerpt: "Essential practices for maintaining soil fertility and improving crop yields sustainably.",
<<<<<<< HEAD
    image: "/soil.png",
=======
    image: "/placeholder.svg?height=100&width=150",
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    readTime: "6 min read",
  },
  {
    id: 4,
    title: "Integrated Pest Management Strategies",
    category: "Hama",
    excerpt: "Effective approaches to control pests while minimizing environmental impact on farms.",
<<<<<<< HEAD
    image: "/pest.png",
=======
    image: "/placeholder.svg?height=100&width=150",
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    readTime: "8 min read",
  },
]

export default function Dashboard() {
  const [selectedStation, setSelectedStation] = useState("wisnu")
  const [activeView, setActiveView] = useState("dashboard")
  const [showPassword, setShowPassword] = useState(false)
  const [calibrationData, setCalibrationData] = useState({
    temperature: { multiplier: 1.0, offset: 0.0 },
    humidity: { multiplier: 1.0, offset: 0.0 },
    light: { multiplier: 1.0, offset: 0.0 },
  })

  // Sensor data
  const sensorData = {
<<<<<<< HEAD
    temperature: { value: 32.1, unit: "°C", color: "text-blue-600", bgColor: "bg-blue-50", icon: Thermometer },
    humidity: { value: 64.7, unit: "RH", color: "text-green-600", bgColor: "bg-green-50", icon: Droplets },
    light: { value: 12780, unit: "Lux", color: "text-orange-600", bgColor: "bg-orange-50", icon: Sun },
    solarCurrent: { value: 35.6, unit: "mA", color: "text-green-600", bgColor: "bg-green-50", icon: Zap },
    solarVoltage: { value: 2.09, unit: "mV", color: "text-orange-600", bgColor: "bg-orange-50", icon: Gauge },
    solarWatt: { value: 0.9, unit: "mW", color: "text-orange-600", bgColor: "bg-orange-50", icon: Zap },
    wind: { value: 1.4, unit: "Knot", color: "text-green-600", bgColor: "bg-green-50", icon: Wind },
=======
    temperature: { value: 35.1, unit: "°C", color: "text-blue-600", bgColor: "bg-blue-50", icon: Thermometer },
    humidity: { value: 57.7, unit: "RH", color: "text-green-600", bgColor: "bg-green-50", icon: Droplets },
    light: { value: 44.17, unit: "Lux", color: "text-orange-600", bgColor: "bg-orange-50", icon: Sun },
    solarCurrent: { value: -0.2, unit: "mA", color: "text-green-600", bgColor: "bg-green-50", icon: Zap },
    solarVoltage: { value: 1.09, unit: "mV", color: "text-orange-600", bgColor: "bg-orange-50", icon: Gauge },
    solarWatt: { value: 0, unit: "mW", color: "text-orange-600", bgColor: "bg-orange-50", icon: Zap },
    wind: { value: 0, unit: "Knot", color: "text-green-600", bgColor: "bg-green-50", icon: Wind },
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    rain: { value: 0, unit: "mm", color: "text-blue-600", bgColor: "bg-blue-50", icon: CloudRain },
  }

  const handleCalibrationUpdate = (sensor: string, field: string, value: number) => {
    setCalibrationData((prev) => ({
      ...prev,
      [sensor]: {
        ...prev[sensor as keyof typeof prev],
        [field]: value,
      },
    }))
  }

  const SensorCard = ({
    title,
    value,
    unit,
    icon: Icon,
    color,
    bgColor,
  }: {
    title: string
    value: number
    unit: string
<<<<<<< HEAD
    icon: LucideIcon
=======
    icon: any
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
    color: string
    bgColor: string
  }) => (
    <Card className={`${bgColor} border-0 hover:shadow-md transition-shadow`}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <p className={`text-2xl font-bold ${color}`}>
              {value} <span className="text-sm font-normal">{unit}</span>
            </p>
          </div>
          <Icon className={`w-8 h-8 ${color}`} />
        </div>
      </CardContent>
    </Card>
  )

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-gray-50">
        {/* Sidebar */}
        <Sidebar className="border-r border-gray-200">
          <SidebarHeader className="border-b border-gray-200 p-4">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 bg-gradient-to-br from-green-600 to-emerald-600 rounded-full flex items-center justify-center">
                <Leaf className="w-5 h-5 text-white" />
              </div>
              <div>
                <h2 className="font-bold text-gray-900">Atamagri App</h2>
                <p className="text-xs text-gray-500">Code: 12345</p>
              </div>
            </div>
          </SidebarHeader>

          <SidebarContent>
            <SidebarGroup>
              <SidebarGroupLabel>Navigation</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton onClick={() => setActiveView("dashboard")} isActive={activeView === "dashboard"}>
                      <BarChart3 className="w-4 h-4" />
                      Dashboard
                    </SidebarMenuButton>
                  </SidebarMenuItem>
<<<<<<< HEAD
                  <SidebarMenuItem>
                    <Link
                      href="/drone-detection"
                      className="flex items-center w-full px-3 py-2 rounded hover:bg-gray-100"
                    >
                      <Airplay className="w-4 h-4 mr-2" />
                      Drone Detection
                    </Link>
                  </SidebarMenuItem>
=======
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            <SidebarGroup>
              <SidebarGroupLabel>Stasiun Cuaca</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => setSelectedStation("wisnu")}
                      isActive={selectedStation === "wisnu"}
                    >
                      <MapPin className="w-4 h-4" />
<<<<<<< HEAD
                      Stasiun Test 1
=======
                      wisnu
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => setSelectedStation("test2")}
                      isActive={selectedStation === "test2"}
                    >
                      <MapPin className="w-4 h-4" />
                      Stasiun Test 2
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>

          <SidebarFooter className="border-t border-gray-200 p-4">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="w-full justify-start">
                  <User className="w-4 h-4 mr-2" />
                  User Settings
                  <ChevronDown className="w-4 h-4 ml-auto" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-56">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={() => setActiveView("settings")}>
                  <Settings className="w-4 h-4 mr-2" />
                  Account Settings
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setActiveView("calibration")}>
                  <Settings className="w-4 h-4 mr-2" />
                  Sensor Calibration
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem>
                  <Link href="/" className="flex items-center w-full">
                    Logout
                  </Link>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarFooter>
        </Sidebar>

        {/* Main Content */}
        <SidebarInset className="flex-1">
          {/* Header */}
          <header className="sticky top-0 z-40 bg-white border-b border-gray-200 px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <SidebarTrigger />
                <div>
                  <h1 className="text-xl font-bold text-gray-900">
                    {activeView === "dashboard" && "Weather Dashboard"}
                    {activeView === "settings" && "Account Settings"}
                    {activeView === "calibration" && "Sensor Calibration"}
                  </h1>
                  <p className="text-sm text-gray-500">
                    Station: {selectedStation} • Last updated: {new Date().toLocaleTimeString()}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-4">
                <Button variant="outline" size="sm">
                  <Download className="w-4 h-4 mr-2" />
                  Download Data
                </Button>
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
                <Button variant="outline" size="sm">
                  <Bell className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </header>

          {/* Dashboard Content */}
          <main className="p-6 space-y-6">
            {activeView === "dashboard" && (
              <>
                {/* Sensor Data Cards */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <SensorCard
<<<<<<< HEAD
                    title="Temperatur"
=======
                    title="Temperature"
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                    value={sensorData.temperature.value}
                    unit={sensorData.temperature.unit}
                    icon={sensorData.temperature.icon}
                    color={sensorData.temperature.color}
                    bgColor={sensorData.temperature.bgColor}
                  />
                  <SensorCard
                    title="Kelembapan"
                    value={sensorData.humidity.value}
                    unit={sensorData.humidity.unit}
                    icon={sensorData.humidity.icon}
                    color={sensorData.humidity.color}
                    bgColor={sensorData.humidity.bgColor}
                  />
                  <SensorCard
                    title="Intensitas Cahaya"
                    value={sensorData.light.value}
                    unit={sensorData.light.unit}
                    icon={sensorData.light.icon}
                    color={sensorData.light.color}
                    bgColor={sensorData.light.bgColor}
                  />
                  <SensorCard
                    title="Arus Solar Cell"
                    value={sensorData.solarCurrent.value}
                    unit={sensorData.solarCurrent.unit}
                    icon={sensorData.solarCurrent.icon}
                    color={sensorData.solarCurrent.color}
                    bgColor={sensorData.solarCurrent.bgColor}
                  />
                  <SensorCard
<<<<<<< HEAD
                    title="Tegangan Solar Cell"
=======
                    title="Tegangan Solar"
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                    value={sensorData.solarVoltage.value}
                    unit={sensorData.solarVoltage.unit}
                    icon={sensorData.solarVoltage.icon}
                    color={sensorData.solarVoltage.color}
                    bgColor={sensorData.solarVoltage.bgColor}
                  />
                  <SensorCard
                    title="Watt Solar Cell"
                    value={sensorData.solarWatt.value}
                    unit={sensorData.solarWatt.unit}
                    icon={sensorData.solarWatt.icon}
                    color={sensorData.solarWatt.color}
                    bgColor={sensorData.solarWatt.bgColor}
                  />
                  <SensorCard
<<<<<<< HEAD
                    title="Angin"
=======
                    title="Wind"
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                    value={sensorData.wind.value}
                    unit={sensorData.wind.unit}
                    icon={sensorData.wind.icon}
                    color={sensorData.wind.color}
                    bgColor={sensorData.wind.bgColor}
                  />
                  <SensorCard
<<<<<<< HEAD
                    title="Curah Hujan"
=======
                    title="Rain Gauge"
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                    value={sensorData.rain.value}
                    unit={sensorData.rain.unit}
                    icon={sensorData.rain.icon}
                    color={sensorData.rain.color}
                    bgColor={sensorData.rain.bgColor}
                  />
                </div>

                {/* Charts */}
                <div className="grid md:grid-cols-2 gap-6">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Thermometer className="w-5 h-5 text-blue-600" />
                        <span>Temperature Trend</span>
                        <Badge variant="secondary" className="ml-auto">
                          Realtime
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={temperatureData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#2563eb" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center space-x-2">
                        <Droplets className="w-5 h-5 text-green-600" />
                        <span>Humidity Trend</span>
                        <Badge variant="secondary" className="ml-auto">
                          Realtime
                        </Badge>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={200}>
                        <LineChart data={humidityData}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="time" />
                          <YAxis />
                          <Tooltip />
                          <Line type="monotone" dataKey="value" stroke="#16a34a" strokeWidth={2} />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>
                </div>

                {/* Weather Forecast */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Sun className="w-5 h-5 text-orange-600" />
                      <span>Prakiraan Cuaca</span>
                    </CardTitle>
                    <CardDescription>24-hour weather forecast</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                      {weatherForecast.map((forecast, index) => (
                        <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                          <p className="text-sm font-medium text-gray-600">{forecast.time}</p>
                          <div className="text-2xl my-2">{forecast.icon}</div>
                          <p className="text-lg font-bold text-gray-900">{forecast.temp}°C</p>
                          <p className="text-xs text-gray-500">{forecast.wind} km/h</p>
                          <p className="text-xs text-blue-600">{forecast.rain}%</p>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                {/* News Articles */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Newspaper className="w-5 h-5 text-purple-600" />
                      <span>Artikel & Informasi Terkini</span>
                    </CardTitle>
                    <CardDescription>Latest agricultural news and farming tips</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid md:grid-cols-2 gap-4">
                      {newsArticles.map((article) => (
                        <div
                          key={article.id}
                          className="flex space-x-4 p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors cursor-pointer"
                        >
<<<<<<< HEAD
                          <Image
                            src={article.image || "/placeholder.svg"}
                            alt={article.title}
                            width={80}
                            height={80}
=======
                          <img
                            src={article.image || "/placeholder.svg"}
                            alt={article.title}
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
                            className="w-20 h-20 object-cover rounded-lg flex-shrink-0"
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-1">
                              <Badge variant="outline" className="text-xs">
                                {article.category}
                              </Badge>
                              <span className="text-xs text-gray-500">{article.readTime}</span>
                            </div>
                            <h4 className="font-semibold text-gray-900 text-sm mb-1 line-clamp-2">{article.title}</h4>
                            <p className="text-xs text-gray-600 line-clamp-2">{article.excerpt}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            {activeView === "calibration" && (
              <Card>
                <CardHeader>
                  <CardTitle>Kalibrasi Sensor</CardTitle>
                  <CardDescription>Adjust sensor calibration constants for accurate readings</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid gap-4">
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div>
                        <Label className="text-sm font-medium">Station ID</Label>
                        <Input value={selectedStation} disabled className="mt-1" />
                      </div>
                      <div>
                        <Label className="text-sm font-medium">Station Name</Label>
                        <div className="flex space-x-2 mt-1">
                          <Input defaultValue="Weather Station 1" />
                          <Button size="sm" className="bg-green-600 hover:bg-green-700">
                            Simpan
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="space-y-4">
                    <h3 className="text-lg font-semibold">Sensor Calibration</h3>

                    {/* Temperature Calibration */}
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-3 flex items-center">
                        <Thermometer className="w-4 h-4 mr-2 text-blue-600" />
                        Temperature Sensor
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <Label className="text-sm">Raw Value</Label>
                          <Input value="34.8" disabled />
                        </div>
                        <div>
                          <Label className="text-sm">Multiplier</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.temperature.multiplier}
                            onChange={(e) =>
                              handleCalibrationUpdate("temperature", "multiplier", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Offset</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.temperature.offset}
                            onChange={(e) =>
                              handleCalibrationUpdate("temperature", "offset", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Calibrated Value</Label>
                          <Input value="35.1 °C" disabled />
                          <Button size="sm" className="mt-2 w-full bg-blue-600 hover:bg-blue-700">
                            Update
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Humidity Calibration */}
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-3 flex items-center">
                        <Droplets className="w-4 h-4 mr-2 text-green-600" />
                        Humidity Sensor
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <Label className="text-sm">Raw Value</Label>
                          <Input value="57.2" disabled />
                        </div>
                        <div>
                          <Label className="text-sm">Multiplier</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.humidity.multiplier}
                            onChange={(e) =>
                              handleCalibrationUpdate("humidity", "multiplier", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Offset</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.humidity.offset}
                            onChange={(e) =>
                              handleCalibrationUpdate("humidity", "offset", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Calibrated Value</Label>
                          <Input value="57.7 RH" disabled />
                          <Button size="sm" className="mt-2 w-full bg-green-600 hover:bg-green-700">
                            Update
                          </Button>
                        </div>
                      </div>
                    </div>

                    {/* Light Sensor Calibration */}
                    <div className="p-4 border rounded-lg">
                      <h4 className="font-medium mb-3 flex items-center">
                        <Sun className="w-4 h-4 mr-2 text-orange-600" />
                        Light Intensity Sensor
                      </h4>
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <Label className="text-sm">Raw Value</Label>
                          <Input value="43.9" disabled />
                        </div>
                        <div>
                          <Label className="text-sm">Multiplier</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.light.multiplier}
                            onChange={(e) =>
                              handleCalibrationUpdate("light", "multiplier", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Offset</Label>
                          <Input
                            type="number"
                            step="0.1"
                            value={calibrationData.light.offset}
                            onChange={(e) =>
                              handleCalibrationUpdate("light", "offset", Number.parseFloat(e.target.value))
                            }
                          />
                        </div>
                        <div>
                          <Label className="text-sm">Calibrated Value</Label>
                          <Input value="44.17 Lux" disabled />
                          <Button size="sm" className="mt-2 w-full bg-orange-600 hover:bg-orange-700">
                            Update
                          </Button>
                        </div>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}

            {activeView === "settings" && (
              <div className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Pengaturan Akun</CardTitle>
                    <CardDescription>Manage your account information and preferences</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    <div className="space-y-4">
                      <h3 className="text-lg font-semibold">User Profile</h3>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="name">Name</Label>
                          <Input id="name" defaultValue="John Doe" />
                        </div>
                        <div>
                          <Label htmlFor="email">Email</Label>
                          <Input id="email" type="email" defaultValue="john.doe@example.com" />
                        </div>
                      </div>
                      <Button className="bg-green-600 hover:bg-green-700">Simpan</Button>
                    </div>

                    <div className="space-y-4 pt-6 border-t">
                      <h3 className="text-lg font-semibold">Password Management</h3>
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div>
                          <Label htmlFor="oldPassword">Old Password</Label>
                          <div className="relative">
                            <Input
                              id="oldPassword"
                              type={showPassword ? "text" : "password"}
                              placeholder="Enter old password"
                            />
                            <button
                              type="button"
                              onClick={() => setShowPassword(!showPassword)}
                              className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-500"
                            >
                              {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                            </button>
                          </div>
                        </div>
                        <div>
                          <Label htmlFor="newPassword">New Password</Label>
                          <Input id="newPassword" type="password" placeholder="Enter new password" />
                        </div>
                        <div>
                          <Label htmlFor="confirmPassword">Confirm New Password</Label>
                          <Input id="confirmPassword" type="password" placeholder="Confirm new password" />
                        </div>
                      </div>
                      <Button className="bg-blue-600 hover:bg-blue-700">Ganti</Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </main>

          {/* Footer */}
          <footer className="border-t border-gray-200 px-6 py-4 bg-white">
<<<<<<< HEAD
            <p className="text-sm text-gray-500 text-center">Copyright 2025 Atamagri. All rights reserved.</p>
=======
            <p className="text-sm text-gray-500 text-center">Copyright © 2025 Atamagri. All rights reserved.</p>
>>>>>>> 9e0c12cf6c0087bef71d83d70afd8326da0a6a21
          </footer>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}
