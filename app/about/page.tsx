import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { Leaf, Users, Target, Award, ArrowRight } from "lucide-react"
import Link from "next/link"
import Image from "next/image"

export default function AboutPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-50">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-md border-b border-green-100">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link href="/" className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-green-600 to-emerald-600 rounded-full flex items-center justify-center">
                <Leaf className="w-6 h-6 text-white" />
              </div>
              <span className="text-2xl font-bold text-green-800">Atamagri</span>
            </Link>

            {/* Desktop Navigation */}
            <nav className="hidden md:flex items-center space-x-8">
              <Link href="/" className="text-gray-700 hover:text-green-600 transition-colors">
                Home
              </Link>
              <Link href="/about" className="text-green-600 font-medium">
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
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-3xl mx-auto text-center">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">Our Mission to Transform Agriculture</h1>
            <p className="text-xl text-gray-600 mb-8">
              Atamagri is dedicated to empowering farmers with climate intelligence and smart technology to increase
              yields, reduce risks, and promote sustainable agriculture across Indonesia.
            </p>
          </div>
        </div>
      </section>

      {/* Our Story */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div>
              <h2 className="text-3xl font-bold text-gray-900 mb-6">Our Story</h2>
              <div className="space-y-4 text-gray-600">
                <p>
                  Atamagri was founded in 2020 by a team of researchers from Universitas Sebelas Maret with expertise in
                  Physics, Statistics, and Agricultural Technology. Our journey began with a simple observation: farmers
                  across Indonesia were increasingly vulnerable to unpredictable weather patterns caused by climate
                  change.
                </p>
                <p>
                  We set out to create affordable, accessible weather monitoring technology that could provide
                  hyperlocal data to farmers, helping them make better decisions about planting, irrigation, and
                  harvesting. What started as a research project quickly evolved into a comprehensive agricultural
                  intelligence platform.
                </p>
                <p>
                  Today, Atamagri serves thousands of farmers, fishermen, and researchers across Indonesia, providing
                  real-time weather data, AI-powered recommendations, and analytics that help increase yields, reduce
                  resource waste, and build climate resilience.
                </p>
              </div>
            </div>
            <div className="relative">
              <div className="bg-gradient-to-br from-green-400 to-emerald-600 rounded-2xl p-6 shadow-2xl">
                <Image
                  src="/placeholder.svg?height=400&width=500"
                  alt="Atamagri Team"
                  width={500}
                  height={400}
                  className="rounded-lg"
                />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-16 bg-gradient-to-br from-green-50 to-emerald-50">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Our Values</h2>
            <p className="text-xl text-gray-600">The principles that guide everything we do</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto">
                  <Target className="w-8 h-8 text-green-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">Innovation</h3>
                <p className="text-gray-600">
                  We continuously develop cutting-edge technology to solve real agricultural challenges.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto">
                  <Users className="w-8 h-8 text-blue-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">Accessibility</h3>
                <p className="text-gray-600">
                  We make advanced agricultural technology affordable and accessible to farmers of all scales.
                </p>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-16 h-16 bg-purple-100 rounded-full flex items-center justify-center mx-auto">
                  <Award className="w-8 h-8 text-purple-600" />
                </div>
                <h3 className="text-xl font-bold text-gray-900">Sustainability</h3>
                <p className="text-gray-600">
                  We promote sustainable farming practices that protect the environment for future generations.
                </p>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-16 bg-white">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-gray-900 mb-4">Meet Our Team</h2>
            <p className="text-xl text-gray-600">The experts behind Atamagri's innovation</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-24 h-24 bg-gray-200 rounded-full mx-auto"></div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Dr. Ahmad Susanto</h3>
                  <p className="text-green-600 font-medium">CEO & Co-Founder</p>
                  <p className="text-sm text-gray-600">PhD in Agricultural Physics</p>
                </div>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-24 h-24 bg-gray-200 rounded-full mx-auto"></div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Sari Wijayanti, M.Sc</h3>
                  <p className="text-green-600 font-medium">CTO & Co-Founder</p>
                  <p className="text-sm text-gray-600">Master in Statistics & Data Science</p>
                </div>
              </CardContent>
            </Card>

            <Card className="text-center p-6">
              <CardContent className="space-y-4">
                <div className="w-24 h-24 bg-gray-200 rounded-full mx-auto"></div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900">Budi Hartono, M.Eng</h3>
                  <p className="text-green-600 font-medium">Head of Engineering</p>
                  <p className="text-sm text-gray-600">Master in Agricultural Technology</p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16 bg-gradient-to-r from-green-600 to-emerald-600 text-white">
        <div className="container mx-auto px-4 text-center">
          <h2 className="text-3xl font-bold mb-4">Ready to Join Our Mission?</h2>
          <p className="text-xl mb-8 opacity-90">
            Be part of the agricultural revolution and help build a more sustainable future.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link href="/dashboard">
              <Button size="lg" className="bg-white text-green-600 hover:bg-gray-100">
                Get Started Today
                <ArrowRight className="ml-2 w-4 h-4" />
              </Button>
            </Link>
            <Link href="/contact">
              <Button
                size="lg"
                variant="outline"
                className="border-white text-white hover:bg-white hover:text-green-600"
              >
                Contact Us
              </Button>
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-900 text-white py-8">
        <div className="container mx-auto px-4 text-center">
          <p className="text-gray-400">Copyright © 2025 Atamagri. All rights reserved.</p>
        </div>
      </footer>
    </div>
  )
}
