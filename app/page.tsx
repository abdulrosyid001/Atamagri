"use client"

import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import {
  Cloud,
  Droplets,
  Smartphone,
  BarChart3,
  Shield,
  Users,
  TrendingUp,
  Star,
  Menu,
  X,
  Play,
  ArrowRight,
  Zap,
  Wifi,
  Brain,
  Leaf,
  Sprout,
} from "lucide-react"
import Image from "next/image"
import Link from "next/link"
import { useState } from "react"

export default function AtamagriLanding() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false)

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-50">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-green-100">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-emerald-600 rounded-full flex items-center justify-center">
                <Leaf className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-green-800">Atamagri</span>
            </div>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-8">
              <Link href="/" className="text-gray-700 hover:text-green-600 transition-colors">
                Home
              </Link>
              <Link href="/about" className="text-gray-700 hover:text-green-600 transition-colors">
                About
              </Link>
              <Link href="/solutions" className="text-gray-700 hover:text-green-600 transition-colors">
                Solutions
              </Link>
              <Link href="/atamastation" className="text-gray-700 hover:text-green-600 transition-colors">
                AtamaStation
              </Link>
              <Link href="/pricing" className="text-gray-700 hover:text-green-600 transition-colors">
                Pricing
              </Link>
              <Link href="/contact" className="text-gray-700 hover:text-green-600 transition-colors">
                Contact
              </Link>
            </nav>

            <div className="hidden md:flex items-center space-x-4">
              <Link href="/login">
                <Button variant="outline" className="border-green-600 text-green-600 hover:bg-green-50">
                  Login
                </Button>
              </Link>
              <Link href="/dashboard">
                <Button className="bg-green-600 hover:bg-green-700 text-white">Access Dashboard</Button>
              </Link>
            </div>

            {/* Mobile Menu Button */}
            <button className="md:hidden" onClick={() => setMobileMenuOpen(!mobileMenuOpen)}>
              {mobileMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
            </button>
          </div>

          {/* Mobile Menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-4 border-t border-green-100">
              <nav className="flex flex-col space-y-4 mt-4">
                <Link href="/" className="text-gray-700 hover:text-green-600 transition-colors">
                  Home
                </Link>
                <Link href="/about" className="text-gray-700 hover:text-green-600 transition-colors">
                  About
                </Link>
                <Link href="/solutions" className="text-gray-700 hover:text-green-600 transition-colors">
                  Solutions
                </Link>
                <Link href="/atamastation" className="text-gray-700 hover:text-green-600 transition-colors">
                  AtamaStation
                </Link>
                <Link href="/pricing" className="text-gray-700 hover:text-green-600 transition-colors">
                  Pricing
                </Link>
                <Link href="/contact" className="text-gray-700 hover:text-green-600 transition-colors">
                  Contact
                </Link>
                <div className="flex flex-col space-y-2 pt-4">
                  <Link href="/login">
                    <Button variant="outline" className="w-full border-green-600 text-green-600 hover:bg-green-50">
                      Login
                    </Button>
                  </Link>
                  <Link href="/dashboard">
                    <Button className="w-full bg-green-600 hover:bg-green-700 text-white">Access Dashboard</Button>
                  </Link>
                </div>
              </nav>
            </div>
          )}
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20 lg:py-32">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8">
              <div className="space-y-4">
                <Badge className="bg-green-100 text-green-800 hover:bg-green-200">
                  🌱 Smart Agriculture Technology
                </Badge>
                <h1 className="text-4xl lg:text-6xl font-bold text-gray-900 leading-tight">
                  Climate Intelligence for <span className="text-green-600">Smart Farming</span>
                </h1>
                <p className="text-xl text-gray-600 leading-relaxed">
                  Real-time weather monitoring, climate data analytics, and AI-driven decision recommendations for
                  farmers, fishermen, and researchers with our AtamaStation IoT platform.
                </p>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">50K+</div>
                  <div className="text-sm text-gray-600">Active Users</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">85%</div>
                  <div className="text-sm text-gray-600">Yield Improvement</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">24/7</div>
                  <div className="text-sm text-gray-600">Real-time Monitoring</div>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row gap-4">
                <Link href="/dashboard">
                  <Button size="lg" className="bg-green-600 hover:bg-green-700 text-white">
                    Access Your Dashboard
                    <ArrowRight className="ml-2 w-4 h-4" />
                  </Button>
                </Link>
                <Button size="lg" variant="outline" className="border-green-600 text-green-600 hover:bg-green-50">
                  <Play className="mr-2 w-4 h-4" />
                  Learn More
                </Button>
              </div>
            </div>

            <div className="relative">
              <div className="bg-gradient-to-br from-green-400 to-emerald-600 rounded-2xl p-8 shadow-2xl">
                <Image
                  src="/dashboard.png"
                  alt="Atamagri Dashboard Preview"
                  width={500}
                  height={400}
                  className="rounded-lg shadow-lg"
                />
              </div>
              <div className="absolute -bottom-6 -right-6 bg-white rounded-xl p-4 shadow-lg">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium">Live Data</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Problem/Solution Section */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Solving Agriculture&apos;s Climate Challenges
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Climate unpredictability threatens food security. Atamagri provides the intelligence farmers need to adapt
              and thrive.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-gray-900">The Challenge</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-red-100 rounded-full flex items-center justify-center mt-1">
                    <TrendingUp className="w-4 h-4 text-red-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Unpredictable Weather</h4>
                    <p className="text-gray-600">Climate change creates extreme weather patterns that damage crops</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-orange-100 rounded-full flex items-center justify-center mt-1">
                    <Droplets className="w-4 h-4 text-orange-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Water Management</h4>
                    <p className="text-gray-600">Inefficient irrigation leads to water waste and poor yields</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-yellow-100 rounded-full flex items-center justify-center mt-1">
                    <Users className="w-4 h-4 text-yellow-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Limited Data Access</h4>
                    <p className="text-gray-600">Farmers lack real-time environmental data for decision making</p>
                  </div>
                </div>
              </div>
            </div>

            <div className="space-y-6">
              <h3 className="text-2xl font-bold text-green-600">Our Solution</h3>
              <div className="space-y-4">
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-green-100 rounded-full flex items-center justify-center mt-1">
                    <Wifi className="w-4 h-4 text-green-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Real-time Monitoring</h4>
                    <p className="text-gray-600">AtamaStation provides continuous environmental data collection</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center mt-1">
                    <Brain className="w-4 h-4 text-blue-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">AI-Powered Insights</h4>
                    <p className="text-gray-600">Machine learning algorithms provide actionable recommendations</p>
                  </div>
                </div>
                <div className="flex items-start space-x-3">
                  <div className="w-6 h-6 bg-purple-100 rounded-full flex items-center justify-center mt-1">
                    <BarChart3 className="w-4 h-4 text-purple-600" />
                  </div>
                  <div>
                    <h4 className="font-semibold text-gray-900">Data Analytics</h4>
                    <p className="text-gray-600">Historical trends and predictive analytics for better planning</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Key Features Overview */}
      <section className="py-20 bg-gradient-to-br from-green-50 to-emerald-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">
              Comprehensive Agricultural Intelligence
            </h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Everything you need to make data-driven decisions for your farming operations
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Cloud className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">Real-Time Weather Data</h3>
                <p className="text-gray-600">
                  Continuous monitoring of temperature, humidity, rainfall, wind, and solar radiation
                </p>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
                  <Brain className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">AI-Powered Support</h3>
                <p className="text-gray-600">
                  Intelligent recommendations for planting, irrigation, and harvest timing
                </p>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-orange-100 rounded-lg flex items-center justify-center">
                  <Zap className="w-6 h-6 text-orange-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">AtamaStation IoT</h3>
                <p className="text-gray-600">Advanced IoT sensors for comprehensive environmental monitoring</p>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">Data Analytics</h3>
                <p className="text-gray-600">Historical trends, predictive modeling, and customizable reports</p>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-teal-100 rounded-lg flex items-center justify-center">
                  <Smartphone className="w-6 h-6 text-teal-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">Web & Mobile Access</h3>
                <p className="text-gray-600">Access your data anywhere with responsive web and mobile applications</p>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white/80 backdrop-blur-sm hover:shadow-lg transition-shadow">
              <CardContent className="space-y-4">
                <div className="w-12 h-12 bg-red-100 rounded-lg flex items-center justify-center">
                  <Shield className="w-6 h-6 text-red-600" />
                </div>
                <h3 className="text-lg font-bold text-gray-900">Early Warning System</h3>
                <p className="text-gray-600">Proactive alerts for extreme weather and optimal farming conditions</p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">How Atamagri Works</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              A simple three-step process to transform your agricultural operations
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-green-600">1</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900">Deploy AtamaStation</h3>
              <p className="text-gray-600">
                Install our IoT weather station on your farm for continuous environmental monitoring
              </p>
            </div>

            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-blue-600">2</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900">Access Your Dashboard</h3>
              <p className="text-gray-600">
                Monitor real-time data, view analytics, and receive AI-powered recommendations
              </p>
            </div>

            <div className="text-center space-y-4">
              <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                <span className="text-2xl font-bold text-purple-600">3</span>
              </div>
              <h3 className="text-xl font-bold text-gray-900">Optimize Operations</h3>
              <p className="text-gray-600">
                Make data-driven decisions to improve yields, reduce costs, and minimize risks
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Target Audience */}
      <section className="py-20 bg-gradient-to-br from-green-50 to-emerald-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">Empowering Agricultural Communities</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Serving farmers, fishermen, and researchers with tailored solutions
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-8 text-center border-green-200 hover:shadow-lg transition-shadow">
              <CardContent className="space-y-6">
                <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                  <Sprout className="w-10 h-10 text-green-600" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-gray-900">Farmers</div>
                </div>
                <p className="text-gray-600">
                  Optimize crop yields with precision agriculture and smart irrigation management
                </p>
              </CardContent>
            </Card>

            <Card className="p-8 text-center border-blue-200 hover:shadow-lg transition-shadow">
              <CardContent className="space-y-6">
                <div className="w-20 h-20 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <Droplets className="w-10 h-10 text-blue-600" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-gray-900">Fishermen</div>
                </div>
                <p className="text-gray-600">
                  Ensure safe operations with accurate weather forecasts and marine conditions
                </p>
              </CardContent>
            </Card>

            <Card className="p-8 text-center border-purple-200 hover:shadow-lg transition-shadow">
              <CardContent className="space-y-6">
                <div className="w-20 h-20 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                  <BarChart3 className="w-10 h-10 text-purple-600" />
                </div>
                <div>
                  <div className="text-lg font-semibold text-gray-900">Researchers</div>
                </div>
                <p className="text-gray-600">
                  Access comprehensive climate data for agricultural and environmental studies
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Testimonials */}
      <section className="py-20 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl lg:text-4xl font-bold text-gray-900 mb-4">Success Stories</h2>
            <p className="text-xl text-gray-600 max-w-3xl mx-auto">
              Real results from farmers using Atamagri technology
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="p-6 bg-white border-green-200">
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-1">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <p className="text-gray-700 italic">
                  &quot;Atamagri helped us increase our rice yield by 45% through better irrigation timing and weather
                  predictions.&quot;
                </p>
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <Users className="w-5 h-5 text-green-600" />
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">Budi Santoso</div>
                    <div className="text-sm text-gray-600">Rice Farmer, Central Java</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white border-blue-200">
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-1">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <p className="text-gray-700 italic">
                  &quot;The weather alerts saved our fishing fleet from dangerous storms. Safety and productivity improved
                  significantly.&quot;
                </p>
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                    <Droplets className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">Ahmad Wijaya</div>
                    <div className="text-sm text-gray-600">Fisherman, East Java</div>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card className="p-6 bg-white border-purple-200">
              <CardContent className="space-y-4">
                <div className="flex items-center space-x-1">
                  {[...Array(5)].map((_, i) => (
                    <Star key={i} className="w-4 h-4 fill-yellow-400 text-yellow-400" />
                  ))}
                </div>
                <p className="text-gray-700 italic">
                  &quot;The comprehensive data from Atamagri has been invaluable for our climate research and agricultural
                  studies.&quot;
                </p>
                <div className="flex items-center space-x-3">
                  <div className="w-10 h-10 bg-purple-100 rounded-full flex items-center justify-center">
                    <BarChart3 className="w-5 h-5 text-purple-600" />
                  </div>
                  <div>
                    <div className="font-semibold text-gray-900">Dr. Sari Indrawati</div>
                    <div className="text-sm text-gray-600">Agricultural Researcher</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-20 bg-gradient-to-r from-green-600 to-emerald-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <div className="max-w-3xl mx-auto space-y-8">
            <h2 className="text-3xl lg:text-4xl font-bold">Ready to Transform Your Agriculture?</h2>
            <p className="text-xl opacity-90">
              Join thousands of farmers who are already using Atamagri to optimize their operations and increase yields
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link href="/dashboard">
                <Button size="lg" className="bg-white text-green-600 hover:bg-gray-100">
                  Access Your Dashboard
                  <ArrowRight className="ml-2 w-4 h-4" />
                </Button>
              </Link>
              <Link href="/contact">
                <Button
                  size="lg"
                  variant="outline"
                  className="border-white text-white hover:bg-white hover:text-green-600"
                >
                  Contact Sales
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-16">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-4 gap-8">
            <div className="space-y-4">
              <div className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-green-600 rounded-full flex items-center justify-center">
                  <Leaf className="w-5 h-5 text-white" />
                </div>
                <span className="text-xl font-bold">Atamagri</span>
              </div>
              <p className="text-gray-400">Climate intelligence for smart farming across Indonesia.</p>
              <div className="space-y-2">
                <p className="text-sm text-gray-400">📧 info@atamagri.com</p>
                <p className="text-sm text-gray-400">📱 +62 812-3456-7890</p>
                <p className="text-sm text-gray-400">📍 Solo, Central Java, Indonesia</p>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">Solutions</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link href="/solutions" className="hover:text-white transition-colors">
                    Platform Overview
                  </Link>
                </li>
                <li>
                  <Link href="/atamastation" className="hover:text-white transition-colors">
                    AtamaStation IoT
                  </Link>
                </li>
                <li>
                  <Link href="/dashboard" className="hover:text-white transition-colors">
                    Dashboard
                  </Link>
                </li>
                <li>
                  <Link href="/pricing" className="hover:text-white transition-colors">
                    Pricing
                  </Link>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-400">
                <li>
                  <Link href="/blog" className="hover:text-white transition-colors">
                    Blog
                  </Link>
                </li>
                <li>
                  <Link href="/faq" className="hover:text-white transition-colors">
                    FAQ
                  </Link>
                </li>
                <li>
                  <Link href="/support" className="hover:text-white transition-colors">
                    Support
                  </Link>
                </li>
                <li>
                  <Link href="/contact" className="hover:text-white transition-colors">
                    Contact
                  </Link>
                </li>
              </ul>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-4">Newsletter</h3>
              <p className="text-gray-400 mb-4">Stay updated with agricultural insights</p>
              <div className="space-y-2">
                <Input
                  type="email"
                  placeholder="Enter your email"
                  className="bg-gray-800 border-gray-700 text-white placeholder-gray-400"
                />
                <Button className="w-full bg-green-600 hover:bg-green-700">Subscribe</Button>
              </div>
            </div>
          </div>

          <div className="border-t border-gray-800 mt-12 pt-8 text-center">
            <p className="text-gray-400 text-sm">Copyright © 2025 Atamagri. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  )
}
