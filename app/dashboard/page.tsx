"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
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
  PlusCircle,
  AlertTriangle,
  CheckCircle,
  Clock,
  FileText,
  Wifi,
  WifiOff,
  Filter,
  Layers,
  Activity,
  Plane,
  Camera,
  Battery,
  Radio,
  Navigation,
  CircuitBoard,
  Shield,
  Target,
} from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from "recharts"
import Link from "next/link"
import { useState } from "react"
import DroneControl from "@/components/drone-control"

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

const lightData = [
  { time: "00:00", value: 0 },
  { time: "04:00", value: 5.2 },
  { time: "08:00", value: 28.7 },
  { time: "12:00", value: 44.17 },
  { time: "16:00", value: 32.5 },
  { time: "20:00", value: 8.3 },
]

const solarCurrentData = [
  { time: "00:00", value: 0 },
  { time: "04:00", value: -0.1 },
  { time: "08:00", value: -0.15 },
  { time: "12:00", value: -0.2 },
  { time: "16:00", value: -0.18 },
  { time: "20:00", value: -0.05 },
]

const solarVoltageData = [
  { time: "00:00", value: 0.2 },
  { time: "04:00", value: 0.5 },
  { time: "08:00", value: 0.85 },
  { time: "12:00", value: 1.09 },
  { time: "16:00", value: 0.95 },
  { time: "20:00", value: 0.4 },
]

const solarWattData = [
  { time: "00:00", value: 0 },
  { time: "04:00", value: 0 },
  { time: "08:00", value: 0 },
  { time: "12:00", value: 0 },
  { time: "16:00", value: 0 },
  { time: "20:00", value: 0 },
]

const windData = [
  { time: "00:00", value: 0 },
  { time: "04:00", value: 2 },
  { time: "08:00", value: 5 },
  { time: "12:00", value: 0 },
  { time: "16:00", value: 3 },
  { time: "20:00", value: 1 },
]

const rainData = [
  { time: "00:00", value: 0 },
  { time: "04:00", value: 0 },
  { time: "08:00", value: 0 },
  { time: "12:00", value: 0 },
  { time: "16:00", value: 0 },
  { time: "20:00", value: 0 },
]

// Weather forecast data
const weatherForecast = [
  { time: "12:00", temp: 35, wind: 8, rain: 10, humidity: 58, icon: "☀️" },
  { time: "15:00", temp: 33, wind: 12, rain: 15, humidity: 62, icon: "⛅" },
  { time: "18:00", temp: 29, wind: 6, rain: 5, humidity: 70, icon: "☀️" },
  { time: "21:00", temp: 26, wind: 4, rain: 0, humidity: 75, icon: "🌙" },
  { time: "00:00", temp: 24, wind: 3, rain: 0, humidity: 80, icon: "🌙" },
  { time: "03:00", temp: 22, wind: 5, rain: 20, humidity: 85, icon: "🌧️" },
]

// News articles data
const newsArticles = [
  {
    id: 1,
    title: "Optimizing Rice Irrigation in Dry Season",
    category: "Tanaman",
    excerpt: "Learn effective water management techniques for rice cultivation during drought periods.",
    image: "/placeholder.svg?height=100&width=150",
    readTime: "5 min read",
  },
  {
    id: 2,
    title: "Weather Patterns and Crop Planning",
    category: "Cuaca",
    excerpt: "Understanding seasonal weather changes for better agricultural planning and decision making.",
    image: "/placeholder.svg?height=100&width=150",
    readTime: "7 min read",
  },
  {
    id: 3,
    title: "Soil Health Management Techniques",
    category: "Tanah",
    excerpt: "Essential practices for maintaining soil fertility and improving crop yields sustainably.",
    image: "/placeholder.svg?height=100&width=150",
    readTime: "6 min read",
  },
  {
    id: 4,
    title: "Integrated Pest Management Strategies",
    category: "Hama",
    excerpt: "Effective approaches to control pests while minimizing environmental impact on farms.",
    image: "/placeholder.svg?height=100&width=150",
    readTime: "8 min read",
  },
]

// Mock stations data
const userStations = [
  {
    id: "wisnu",
    name: "wisnu",
    location: "Solo",
    status: "active",
    lastUpdate: "2 minutes ago",
    sensors: {
      temperature: { value: 35.1, unit: "°C", status: "normal" },
      humidity: { value: 57.7, unit: "RH", status: "normal" },
      light: { value: 44.17, unit: "Lux", status: "normal" },
      solarCurrent: { value: -0.2, unit: "mA", status: "normal" },
      solarVoltage: { value: 1.09, unit: "mV", status: "normal" },
      solarWatt: { value: 0, unit: "mW", status: "normal" },
      wind: { value: 0, unit: "Knot", status: "normal" },
      rain: { value: 0, unit: "mm", status: "normal" },
    },
    uptime: "99.2%",
    updateInterval: "30 seconds",
    sensorsNormal: "8/8",
  },
  {
    id: "test2",
    name: "Stasiun Test 2",
    location: "Jakarta",
    status: "active",
    lastUpdate: "5 minutes ago",
    sensors: {
      temperature: { value: 33.8, unit: "°C", status: "warning" },
      humidity: { value: 62.3, unit: "RH", status: "normal" },
      light: { value: 38.5, unit: "Lux", status: "normal" },
      solarCurrent: { value: -0.18, unit: "mA", status: "normal" },
      solarVoltage: { value: 0.95, unit: "mV", status: "normal" },
      solarWatt: { value: 0, unit: "mW", status: "normal" },
      wind: { value: 2, unit: "Knot", status: "normal" },
      rain: { value: 0.5, unit: "mm", status: "normal" },
    },
    uptime: "98.7%",
    updateInterval: "30 seconds",
    sensorsNormal: "7/8",
  },
  {
    id: "station3",
    name: "Stasiun 1 - Jakarta",
    location: "Jakarta",
    status: "inactive",
    lastUpdate: "2 days ago",
    sensors: {
      temperature: { value: 0, unit: "°C", status: "error" },
      humidity: { value: 0, unit: "RH", status: "error" },
      light: { value: 0, unit: "Lux", status: "error" },
      solarCurrent: { value: 0, unit: "mA", status: "error" },
      solarVoltage: { value: 0, unit: "mV", status: "error" },
      solarWatt: { value: 0, unit: "mW", status: "error" },
      wind: { value: 0, unit: "Knot", status: "error" },
      rain: { value: 0, unit: "mm", status: "error" },
    },
    uptime: "0%",
    updateInterval: "N/A",
    sensorsNormal: "0/8",
  },
]

// Mock drone data
const droneData = {
  id: "tello-001",
  name: "DJI Tello",
  status: "connected", // connected, disconnected, flying, idle
  battery: 85,
  signal: 92,
  altitude: 0,
  speed: 0,
  flightTime: "00:00:00",
  lastMission: "Field Survey - Zone A",
  location: "Solo, Central Java",
  isFlying: false,
  diseaseDetections: 15,
  areasScanned: "2.5 ha",
  lastDetection: {
    plant: "Tomato",
    condition: "Late blight",
    confidence: 87.5,
    timestamp: "2 minutes ago"
  }
}

export default function Dashboard() {
  const [selectedStation, setSelectedStation] = useState<string | null>(null)
  const [activeView, setActiveView] = useState("dashboard")
  const [showPassword, setShowPassword] = useState(false)
  const [calibrationData, setCalibrationData] = useState({
    temperature: { multiplier: 1.0, offset: 0.0 },
    humidity: { multiplier: 1.0, offset: 0.0 },
    light: { multiplier: 1.0, offset: 0.0 },
  })
  const [dateRange, setDateRange] = useState("last7days")
  const [exportFormat, setExportFormat] = useState("csv")

  // Calculate summary data
  const activeStationsCount = userStations.filter((station) => station.status === "active").length
  const avgTemperature =
    userStations
      .filter((station) => station.status === "active")
      .reduce((sum, station) => sum + station.sensors.temperature.value, 0) / activeStationsCount
  const avgHumidity =
    userStations
      .filter((station) => station.status === "active")
      .reduce((sum, station) => sum + station.sensors.humidity.value, 0) / activeStationsCount

  const handleCalibrationUpdate = (sensor: string, field: string, value: number) => {
    setCalibrationData((prev) => ({
      ...prev,
      [sensor]: {
        ...prev[sensor as keyof typeof prev],
        [field]: value,
      },
    }))
  }

  const handleExportData = () => {
    alert(`Exporting ${dateRange} data in ${exportFormat.toUpperCase()} format`)
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case "normal":
        return "text-green-600"
      case "warning":
        return "text-amber-600"
      case "error":
        return "text-red-600"
      default:
        return "text-gray-600"
    }
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "normal":
        return <CheckCircle className="w-4 h-4 text-green-600" />
      case "warning":
        return <AlertTriangle className="w-4 h-4 text-amber-600" />
      case "error":
        return <AlertTriangle className="w-4 h-4 text-red-600" />
      default:
        return null
    }
  }

  const getStatusBg = (status: string) => {
    switch (status) {
      case "normal":
        return "bg-green-50"
      case "warning":
        return "bg-amber-50"
      case "error":
        return "bg-red-50"
      default:
        return "bg-gray-50"
    }
  }

  const SensorCard = ({
    title,
    value,
    unit,
    icon: Icon,
    color,
    bgColor,
    status,
    data,
  }: {
    title: string
    value: number
    unit: string
    icon: any
    color: string
    bgColor: string
    status: string
    data?: any[]
  }) => (
    <Card className={`${bgColor} border-0 hover:shadow-md transition-shadow`}>
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-2">
          <div>
            <p className="text-sm font-medium text-gray-600">{title}</p>
            <div className="flex items-center">
              <p className={`text-2xl font-bold ${color}`}>
                {value} <span className="text-sm font-normal">{unit}</span>
              </p>
              <div className="ml-2">{getStatusIcon(status)}</div>
            </div>
          </div>
          <Icon className={`w-8 h-8 ${color}`} />
        </div>
        {data && (
          <div className="h-10 mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={data} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
                <Area type="monotone" dataKey="value" stroke={color.replace("text-", "stroke-")} fill={bgColor} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        )}
        <div className="flex items-center justify-between mt-2">
          <span className="text-xs text-gray-500">Status:</span>
          <span className={`text-xs font-medium ${getStatusColor(status)}`}>{status.toUpperCase()}</span>
        </div>
      </CardContent>
    </Card>
  )

  const StationCard = ({ station }: { station: any }) => (
    <Card
      className={`hover:shadow-md transition-shadow cursor-pointer ${
        station.status === "active" ? "border-green-200" : "border-gray-200"
      }`}
      onClick={() => setSelectedStation(station.id)}
    >
      <CardContent className="p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center">
            <div
              className={`w-3 h-3 rounded-full mr-2 ${
                station.status === "active" ? "bg-green-500 animate-pulse" : "bg-gray-400"
              }`}
            ></div>
            <h3 className="font-medium text-gray-900">{station.name}</h3>
          </div>
          <Badge variant={station.status === "active" ? "default" : "secondary"}>
            {station.status === "active" ? "Active" : "Inactive"}
          </Badge>
        </div>
        <div className="grid grid-cols-2 gap-2 mb-3">
          <div className="text-center p-2 bg-gray-50 rounded">
            <p className="text-xs text-gray-500">Temperature</p>
            <p className="font-semibold text-gray-900">
              {station.status === "active" ? `${station.sensors.temperature.value}°C` : "N/A"}
            </p>
          </div>
          <div className="text-center p-2 bg-gray-50 rounded">
            <p className="text-xs text-gray-500">Humidity</p>
            <p className="font-semibold text-gray-900">
              {station.status === "active" ? `${station.sensors.humidity.value}%` : "N/A"}
            </p>
          </div>
        </div>
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center">
            <MapPin className="w-3 h-3 mr-1" />
            {station.location}
          </div>
          <div className="flex items-center">
            <Clock className="w-3 h-3 mr-1" />
            {station.lastUpdate}
          </div>
        </div>
      </CardContent>
    </Card>
  )

  // Get current station data
  const currentStation = selectedStation ? userStations.find((station) => station.id === selectedStation) : null

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
                    <SidebarMenuButton
                      onClick={() => {
                        setActiveView("dashboard")
                        setSelectedStation(null)
                      }}
                      isActive={activeView === "dashboard" && !selectedStation}
                    >
                      <BarChart3 className="w-4 h-4" />
                      Dashboard
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                  <SidebarMenuItem>
                    <SidebarMenuButton
                      onClick={() => {
                        setActiveView("drone")
                        setSelectedStation(null)
                      }}
                      isActive={activeView === "drone"}
                    >
                      <Plane className="w-4 h-4" />
                      Drone Control
                      <Badge 
                        variant={droneData.status === "connected" ? "default" : "secondary"}
                        className="ml-auto text-xs"
                      >
                        {droneData.status === "connected" ? "Online" : "Offline"}
                      </Badge>
                    </SidebarMenuButton>
                  </SidebarMenuItem>
                </SidebarMenu>
              </SidebarGroupContent>
            </SidebarGroup>

            <SidebarGroup>
              <SidebarGroupLabel>Stasiun Cuaca</SidebarGroupLabel>
              <SidebarGroupContent>
                <SidebarMenu>
                  {userStations.map((station) => (
                    <SidebarMenuItem key={station.id}>
                      <SidebarMenuButton
                        onClick={() => {
                          setSelectedStation(station.id)
                          setActiveView("dashboard")
                        }}
                        isActive={selectedStation === station.id}
                      >
                        <div className="flex items-center">
                          {station.status === "active" ? (
                            <Wifi className="w-4 h-4 mr-2" />
                          ) : (
                            <WifiOff className="w-4 h-4 mr-2" />
                          )}
                          {station.name}
                        </div>
                      </SidebarMenuButton>
                    </SidebarMenuItem>
                  ))}
                  <SidebarMenuItem>
                    <SidebarMenuButton>
                      <PlusCircle className="w-4 h-4 mr-2" />
                      Tambah Stasiun
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
                    {activeView === "dashboard" && !selectedStation && "Dashboard Overview"}
                    {activeView === "dashboard" && selectedStation && currentStation?.name}
                    {activeView === "settings" && "Account Settings"}
                    {activeView === "calibration" && "Sensor Calibration"}
                  </h1>
                  {selectedStation && currentStation && (
                    <p className="text-sm text-gray-500">
                      Location: {currentStation.location} • Last updated: {currentStation.lastUpdate}
                    </p>
                  )}
                </div>
              </div>
              <div className="flex items-center space-x-4">
                {selectedStation && (
                  <>
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="outline" size="sm">
                          <Download className="w-4 h-4 mr-2" />
                          Download Data
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent>
                        <DropdownMenuLabel>Export Options</DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <div className="p-2">
                          <Label htmlFor="date-range" className="text-xs mb-1 block">
                            Date Range
                          </Label>
                          <Select value={dateRange} onValueChange={setDateRange}>
                            <SelectTrigger id="date-range" className="w-full">
                              <SelectValue placeholder="Select date range" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="today">Today</SelectItem>
                              <SelectItem value="last7days">Last 7 days</SelectItem>
                              <SelectItem value="last30days">Last 30 days</SelectItem>
                              <SelectItem value="custom">Custom range</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="p-2">
                          <Label htmlFor="format" className="text-xs mb-1 block">
                            Format
                          </Label>
                          <Select value={exportFormat} onValueChange={setExportFormat}>
                            <SelectTrigger id="format" className="w-full">
                              <SelectValue placeholder="Select format" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="csv">CSV</SelectItem>
                              <SelectItem value="excel">Excel</SelectItem>
                              <SelectItem value="json">JSON</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <DropdownMenuSeparator />
                        <div className="p-2">
                          <Button className="w-full" onClick={handleExportData}>
                            Export
                          </Button>
                        </div>
                      </DropdownMenuContent>
                    </DropdownMenu>
                    <Button variant="outline" size="sm">
                      <RefreshCw className="w-4 h-4 mr-2" />
                      Refresh
                    </Button>
                  </>
                )}
                <Button variant="outline" size="sm">
                  <Bell className="w-4 h-4" />
                </Button>
              </div>
            </div>
          </header>

          {/* Dashboard Content */}
          <main className="p-6 space-y-6">
            {activeView === "dashboard" && !selectedStation && (
              <>
                {/* Network Summary */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <Card className="bg-white">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Total Stations</p>
                          <p className="text-2xl font-bold text-gray-900">{userStations.length}</p>
                        </div>
                        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                          <Layers className="w-5 h-5 text-blue-600" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Active Stations</p>
                          <p className="text-2xl font-bold text-green-600">{activeStationsCount}</p>
                        </div>
                        <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                          <Wifi className="w-5 h-5 text-green-600" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Avg. Temperature</p>
                          <p className="text-2xl font-bold text-blue-600">
                            {avgTemperature.toFixed(1)} <span className="text-sm font-normal">°C</span>
                          </p>
                        </div>
                        <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                          <Thermometer className="w-5 h-5 text-blue-600" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-medium text-gray-600">Avg. Humidity</p>
                          <p className="text-2xl font-bold text-green-600">
                            {avgHumidity.toFixed(1)} <span className="text-sm font-normal">RH</span>
                          </p>
                        </div>
                        <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                          <Droplets className="w-5 h-5 text-green-600" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Drone Status Section */}
                <Card className="bg-gradient-to-r from-purple-50 to-blue-50 border-purple-200">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Plane className="w-6 h-6 text-purple-600" />
                      <span>Drone Operations</span>
                      <Badge variant={droneData.status === "connected" ? "default" : "secondary"}>
                        {droneData.status === "connected" ? "Online" : "Offline"}
                      </Badge>
                    </CardTitle>
                    <CardDescription>AI-powered agricultural surveillance and disease detection</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                      <div className="text-center p-3 bg-white rounded-lg shadow-sm">
                        <Battery className="w-6 h-6 text-green-600 mx-auto mb-2" />
                        <p className="text-sm font-medium text-gray-600">Battery</p>
                        <p className="text-lg font-bold text-gray-900">{droneData.battery}%</p>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg shadow-sm">
                        <Radio className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                        <p className="text-sm font-medium text-gray-600">Signal</p>
                        <p className="text-lg font-bold text-gray-900">{droneData.signal}%</p>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg shadow-sm">
                        <Target className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                        <p className="text-sm font-medium text-gray-600">Detections</p>
                        <p className="text-lg font-bold text-gray-900">{droneData.diseaseDetections}</p>
                      </div>
                      <div className="text-center p-3 bg-white rounded-lg shadow-sm">
                        <CircuitBoard className="w-6 h-6 text-orange-600 mx-auto mb-2" />
                        <p className="text-sm font-medium text-gray-600">Area Scanned</p>
                        <p className="text-lg font-bold text-gray-900">{droneData.areasScanned}</p>
                      </div>
                    </div>
                    
                    {droneData.lastDetection && (
                      <div className="bg-white p-4 rounded-lg shadow-sm border-l-4 border-orange-500">
                        <h4 className="font-semibold text-gray-900 mb-2">Latest Disease Detection</h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-sm">
                          <div>
                            <span className="text-gray-500">Plant:</span>
                            <span className="ml-1 font-medium">{droneData.lastDetection.plant}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Condition:</span>
                            <span className="ml-1 font-medium text-orange-600">{droneData.lastDetection.condition}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Confidence:</span>
                            <span className="ml-1 font-medium">{droneData.lastDetection.confidence}%</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Time:</span>
                            <span className="ml-1 font-medium">{droneData.lastDetection.timestamp}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    
                    <div className="flex gap-2 mt-4">
                      <Button 
                        className="bg-purple-600 hover:bg-purple-700"
                        onClick={() => setActiveView("drone")}
                      >
                        <Plane className="w-4 h-4 mr-2" />
                        Launch Drone Control
                      </Button>
                      <Button variant="outline">
                        <Camera className="w-4 h-4 mr-2" />
                        View Mission History
                      </Button>
                      <Button variant="outline">
                        <Shield className="w-4 h-4 mr-2" />
                        Safety Settings
                      </Button>
                    </div>
                  </CardContent>
                </Card>

                {/* Quick Actions */}
                <div className="flex flex-wrap gap-2">
                  <Button size="sm" className="bg-green-600 hover:bg-green-700">
                    <PlusCircle className="w-4 h-4 mr-2" />
                    Add New Station
                  </Button>
                  <Button size="sm" variant="outline">
                    <Filter className="w-4 h-4 mr-2" />
                    Filter Stations
                  </Button>
                </div>

                {/* Stations List */}
                <div>
                  <h2 className="text-lg font-semibold mb-4">Your Stations</h2>
                  <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {userStations.map((station) => (
                      <StationCard key={station.id} station={station} />
                    ))}
                  </div>
                </div>

                {/* Map View (Placeholder) */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <MapPin className="w-5 h-5 mr-2" />
                      Stations Map
                    </CardTitle>
                    <CardDescription>Geographical distribution of your weather stations</CardDescription>
                  </CardHeader>
                  <CardContent className="h-[300px] bg-gray-100 flex items-center justify-center">
                    <div className="text-center">
                      <MapPin className="w-8 h-8 text-gray-400 mx-auto mb-2" />
                      <p className="text-gray-500">Map view of your stations would appear here</p>
                    </div>
                  </CardContent>
                </Card>
              </>
            )}

            {activeView === "dashboard" && selectedStation && currentStation && (
              <>
                {/* Station Health Indicators */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <Card className="bg-white">
                    <CardContent className="p-4 flex items-center justify-between">
                      <div className="flex items-center">
                        <Activity className="w-5 h-5 text-green-600 mr-3" />
                        <div>
                          <p className="text-sm font-medium text-gray-600">Sensors Status</p>
                          <p className="text-lg font-bold text-gray-900">
                            {currentStation.sensorsNormal} Sensors Normal
                          </p>
                        </div>
                      </div>
                      {currentStation.sensorsNormal === "8/8" ? (
                        <CheckCircle className="w-5 h-5 text-green-600" />
                      ) : (
                        <AlertTriangle className="w-5 h-5 text-amber-600" />
                      )}
                    </CardContent>
                  </Card>

                  <Card className="bg-white">
                    <CardContent className="p-4 flex items-center">
                      <Wifi className="w-5 h-5 text-blue-600 mr-3" />
                      <div>
                        <p className="text-sm font-medium text-gray-600">Uptime</p>
                        <p className="text-lg font-bold text-gray-900">{currentStation.uptime}</p>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-white">
                    <CardContent className="p-4 flex items-center">
                      <Clock className="w-5 h-5 text-purple-600 mr-3" />
                      <div>
                        <p className="text-sm font-medium text-gray-600">Update Interval</p>
                        <p className="text-lg font-bold text-gray-900">{currentStation.updateInterval}</p>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Station Action Buttons */}
                <div className="flex flex-wrap gap-2 mb-6">
                  <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                    <Download className="w-4 h-4 mr-2" />
                    Export Data
                  </Button>
                  <Button size="sm" className="bg-amber-600 hover:bg-amber-700">
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    Configure Alerts
                  </Button>
                  <Button
                    size="sm"
                    className="bg-green-600 hover:bg-green-700"
                    onClick={() => setActiveView("calibration")}
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    Calibrate Sensors
                  </Button>
                  <Button size="sm" className="bg-purple-600 hover:bg-purple-700">
                    <FileText className="w-4 h-4 mr-2" />
                    View Reports
                  </Button>
                </div>

                {/* Tabs for different views */}
                <Tabs defaultValue="sensors" className="mb-6">
                  <TabsList>
                    <TabsTrigger value="sensors">Sensors</TabsTrigger>
                    <TabsTrigger value="forecast">Weather Forecast</TabsTrigger>
                    <TabsTrigger value="articles">Articles</TabsTrigger>
                    <TabsTrigger value="drone">Drone Control</TabsTrigger>
                  </TabsList>
                  <TabsContent value="sensors" className="space-y-6">
                    {/* Sensor Data Cards */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                      <SensorCard
                        title="Temperature"
                        value={currentStation.sensors.temperature.value}
                        unit="°C"
                        icon={Thermometer}
                        color="text-blue-600"
                        bgColor="bg-blue-50"
                        status={currentStation.sensors.temperature.status}
                        data={temperatureData}
                      />
                      <SensorCard
                        title="Kelembapan"
                        value={currentStation.sensors.humidity.value}
                        unit="RH"
                        icon={Droplets}
                        color="text-green-600"
                        bgColor="bg-green-50"
                        status={currentStation.sensors.humidity.status}
                        data={humidityData}
                      />
                      <SensorCard
                        title="Intensitas Cahaya"
                        value={currentStation.sensors.light.value}
                        unit="Lux"
                        icon={Sun}
                        color="text-orange-600"
                        bgColor="bg-orange-50"
                        status={currentStation.sensors.light.status}
                        data={lightData}
                      />
                      <SensorCard
                        title="Arus Solar Cell"
                        value={currentStation.sensors.solarCurrent.value}
                        unit="mA"
                        icon={Zap}
                        color="text-green-600"
                        bgColor="bg-green-50"
                        status={currentStation.sensors.solarCurrent.status}
                        data={solarCurrentData}
                      />
                      <SensorCard
                        title="Tegangan Solar"
                        value={currentStation.sensors.solarVoltage.value}
                        unit="mV"
                        icon={Gauge}
                        color="text-orange-600"
                        bgColor="bg-orange-50"
                        status={currentStation.sensors.solarVoltage.status}
                        data={solarVoltageData}
                      />
                      <SensorCard
                        title="Watt Solar Cell"
                        value={currentStation.sensors.solarWatt.value}
                        unit="mW"
                        icon={Zap}
                        color="text-orange-600"
                        bgColor="bg-orange-50"
                        status={currentStation.sensors.solarWatt.status}
                        data={solarWattData}
                      />
                      <SensorCard
                        title="Wind"
                        value={currentStation.sensors.wind.value}
                        unit="Knot"
                        icon={Wind}
                        color="text-green-600"
                        bgColor="bg-green-50"
                        status={currentStation.sensors.wind.status}
                        data={windData}
                      />
                      <SensorCard
                        title="Rain Gauge"
                        value={currentStation.sensors.rain.value}
                        unit="mm"
                        icon={CloudRain}
                        color="text-blue-600"
                        bgColor="bg-blue-50"
                        status={currentStation.sensors.rain.status}
                        data={rainData}
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
                  </TabsContent>

                  <TabsContent value="forecast">
                    {/* Weather Forecast */}
                    <Card>
                      <CardHeader>
                        <CardTitle className="flex items-center space-x-2">
                          <Sun className="w-5 h-5 text-orange-600" />
                          <span>Prakiraan Cuaca</span>
                        </CardTitle>
                        <CardDescription>24-hour weather forecast for {currentStation.location}</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <div className="grid grid-cols-3 md:grid-cols-6 gap-4">
                          {weatherForecast.map((forecast, index) => (
                            <div key={index} className="text-center p-3 bg-gray-50 rounded-lg">
                              <p className="text-sm font-medium text-gray-600">{forecast.time}</p>
                              <div className="text-2xl my-2">{forecast.icon}</div>
                              <p className="text-lg font-bold text-gray-900">{forecast.temp}°C</p>
                              <div className="grid grid-cols-2 gap-1 mt-2">
                                <div className="text-xs text-gray-500">
                                  <Wind className="w-3 h-3 inline mr-1" />
                                  {forecast.wind} km/h
                                </div>
                                <div className="text-xs text-blue-600">
                                  <Droplets className="w-3 h-3 inline mr-1" />
                                  {forecast.humidity}%
                                </div>
                              </div>
                              <div className="text-xs text-blue-600 mt-1">
                                <CloudRain className="w-3 h-3 inline mr-1" />
                                {forecast.rain}%
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="articles">
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
                              <img
                                src={article.image || "/placeholder.svg"}
                                alt={article.title}
                                className="w-20 h-20 object-cover rounded-lg flex-shrink-0"
                              />
                              <div className="flex-1 min-w-0">
                                <div className="flex items-center space-x-2 mb-1">
                                  <Badge variant="outline" className="text-xs">
                                    {article.category}
                                  </Badge>
                                  <span className="text-xs text-gray-500">{article.readTime}</span>
                                </div>
                                <h4 className="font-semibold text-gray-900 text-sm mb-1 line-clamp-2">
                                  {article.title}
                                </h4>
                                <p className="text-xs text-gray-600 line-clamp-2">{article.excerpt}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </CardContent>
                    </Card>
                  </TabsContent>

                  <TabsContent value="drone" className="mt-4">
                    <Card>
                      <CardContent className="pt-6">
                        <DroneControl />
                      </CardContent>
                    </Card>
                  </TabsContent>
                </Tabs>
              </>
            )}

            {activeView === "drone" && (
              <div className="space-y-6">
                {/* Drone Header */}
                <div className="bg-gradient-to-r from-purple-600 to-blue-600 text-white p-6 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <h2 className="text-2xl font-bold mb-2">Drone Control Center</h2>
                      <p className="text-purple-100">AI-powered agricultural surveillance and disease detection</p>
                    </div>
                    <div className="text-right">
                      <p className="text-purple-100">Status</p>
                      <div className="flex items-center mt-1">
                        <div className={`w-3 h-3 rounded-full mr-2 ${
                          droneData.status === "connected" ? "bg-green-400 animate-pulse" : "bg-red-400"
                        }`}></div>
                        <span className="font-medium capitalize">{droneData.status}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Drone Stats Grid */}
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                  <Card>
                    <CardContent className="p-4 text-center">
                      <Battery className="w-8 h-8 text-green-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Battery</p>
                      <p className="text-xl font-bold">{droneData.battery}%</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <Radio className="w-8 h-8 text-blue-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Signal</p>
                      <p className="text-xl font-bold">{droneData.signal}%</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <Navigation className="w-8 h-8 text-purple-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Altitude</p>
                      <p className="text-xl font-bold">{droneData.altitude}m</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <Gauge className="w-8 h-8 text-orange-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Speed</p>
                      <p className="text-xl font-bold">{droneData.speed} km/h</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <Target className="w-8 h-8 text-red-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Detections</p>
                      <p className="text-xl font-bold">{droneData.diseaseDetections}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardContent className="p-4 text-center">
                      <CircuitBoard className="w-8 h-8 text-teal-600 mx-auto mb-2" />
                      <p className="text-sm text-gray-600">Area</p>
                      <p className="text-xl font-bold">{droneData.areasScanned}</p>
                    </CardContent>
                  </Card>
                </div>

                {/* Main Drone Control Interface */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Plane className="w-5 h-5" />
                      <span>Live Drone Control</span>
                    </CardTitle>
                    <CardDescription>Real-time drone operation and disease detection</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <DroneControl />
                  </CardContent>
                </Card>

                {/* Mission History */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <FileText className="w-5 h-5" />
                      <span>Recent Missions</span>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-3">
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <div>
                            <p className="font-medium">Field Survey - Zone A</p>
                            <p className="text-sm text-gray-500">Completed • 15 detections • 2.5 ha</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-500">2 hours ago</p>
                          <Button variant="outline" size="sm">View Report</Button>
                        </div>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                          <div>
                            <p className="font-medium">Perimeter Check - North Field</p>
                            <p className="text-sm text-gray-500">Completed • 8 detections • 1.8 ha</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-500">1 day ago</p>
                          <Button variant="outline" size="sm">View Report</Button>
                        </div>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
                          <div>
                            <p className="font-medium">Emergency Survey - Zone C</p>
                            <p className="text-sm text-gray-500">In Progress • 23 detections • 3.2 ha</p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-gray-500">3 days ago</p>
                          <Button variant="outline" size="sm">View Report</Button>
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
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
                        <Input value={selectedStation || ""} disabled className="mt-1" />
                      </div>
                      <div>
                        <Label className="text-sm font-medium">Station Name</Label>
                        <div className="flex space-x-2 mt-1">
                          <Input defaultValue={currentStation?.name || "Weather Station"} />
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

                    <div className="space-y-4 pt-6 border-t">
                      <h3 className="text-lg font-semibold">Device Management</h3>
                      <div className="space-y-4">
                        <div className="flex items-center justify-between p-4 border rounded-lg">
                          <div className="flex items-center">
                            <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center mr-4">
                              <Wifi className="w-5 h-5 text-green-600" />
                            </div>
                            <div>
                              <h4 className="font-medium">AtamaStation #12345</h4>
                              <p className="text-sm text-gray-500">Firmware v2.1.0 • Last connected: 2 minutes ago</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">
                            Manage
                          </Button>
                        </div>
                        <div className="flex items-center justify-between p-4 border rounded-lg">
                          <div className="flex items-center">
                            <div className="w-10 h-10 bg-amber-100 rounded-full flex items-center justify-center mr-4">
                              <AlertTriangle className="w-5 h-5 text-amber-600" />
                            </div>
                            <div>
                              <h4 className="font-medium">AtamaStation #67890</h4>
                              <p className="text-sm text-gray-500">Firmware v2.0.5 • Last connected: 3 days ago</p>
                            </div>
                          </div>
                          <Button variant="outline" size="sm">
                            Manage
                          </Button>
                        </div>
                      </div>
                      <Button className="bg-green-600 hover:bg-green-700">
                        <PlusCircle className="w-4 h-4 mr-2" />
                        Add New Device
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </main>

          {/* Footer */}
          <footer className="border-t border-gray-200 px-6 py-4 bg-white">
            <p className="text-sm text-gray-500 text-center">Copyright © 2025 Atamagri. All rights reserved.</p>
          </footer>
        </SidebarInset>
      </div>
    </SidebarProvider>
  )
}
