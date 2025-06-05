# 🌱 Smart Agriculture Dashboard

<div align="center">
  <img src="./public/next.svg" alt="Smart Agriculture Dashboard" width="120" height="120"/>
  
  <p align="center">
    <strong>A modern, responsive web application for smart agriculture monitoring and management</strong>
  </p>

  <p align="center">
    <a href="#-features">Features</a> •
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-documentation">Documentation</a> •
    <a href="#-contributing">Contributing</a>
  </p>

  <p align="center">
    <img src="https://img.shields.io/badge/Next.js-15.2.4-black?style=for-the-badge&logo=next.js" alt="Next.js"/>
    <img src="https://img.shields.io/badge/React-19.0.0-blue?style=for-the-badge&logo=react" alt="React"/>
    <img src="https://img.shields.io/badge/TypeScript-5.0-blue?style=for-the-badge&logo=typescript" alt="TypeScript"/>
    <img src="https://img.shields.io/badge/Tailwind-4.1.8-38B2AC?style=for-the-badge&logo=tailwind-css" alt="Tailwind CSS"/>
  </p>
</div>

---

## 🚀 Features

### Core Functionality
- **📊 Real-time Dashboard** - Monitor environmental conditions and system metrics
- **🌡️ Environmental Monitoring** - Track temperature, humidity, soil moisture, and more
- **📱 Responsive Design** - Optimized for desktop, tablet, and mobile devices
- **🔐 Authentication System** - Secure login and user management
- **📈 Data Visualization** - Interactive charts and graphs with Recharts
- **⚡ Performance Optimized** - Built with Next.js 15 and Turbopack for lightning-fast development

### Technical Highlights
- **Modern UI Components** - Built with shadcn/ui and Radix UI primitives
- **Type-Safe Development** - Full TypeScript support with strict type checking
- **Accessible Design** - ARIA-compliant components following web accessibility standards
- **Dark/Light Mode** - Seamless theme switching (if implemented)
- **Server-Side Rendering** - Optimal performance with Next.js App Router

---

## 🛠️ Tech Stack

| Category | Technology | Version |
|----------|------------|---------|
| **Framework** | Next.js | 15.2.4 |
| **Runtime** | React | 19.0.0 |
| **Language** | TypeScript | 5.0 |
| **Styling** | Tailwind CSS | 4.1.8 |
| **UI Components** | shadcn/ui + Radix UI | Latest |
| **Icons** | Lucide React | 0.513.0 |
| **Charts** | Recharts | 2.15.3 |
| **Build Tool** | Turbopack | Built-in |

---

## 🚀 Quick Start

### Prerequisites

Before you begin, ensure you have the following installed:
- **Node.js** (v18.0.0 or higher) - [Download](https://nodejs.org/)
- **npm** (v9.0.0 or higher) or **yarn** or **pnpm**
- **Git** - [Download](https://git-scm.com/)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd my-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   # or
   yarn install
   # or
   pnpm install
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env.local
   ```
   Edit `.env.local` with your configuration:
   ```env
   NEXT_PUBLIC_API_URL=your_api_endpoint
   DATABASE_URL=your_database_url
   NEXTAUTH_SECRET=your_nextauth_secret
   ```

4. **Run the development server**
   ```bash
   npm run dev
   # or
   yarn dev
   # or
   pnpm dev
   ```

5. **Open your browser**
   
   Navigate to [http://localhost:3000](http://localhost:3000) to see the application running.

---

## 📁 Project Structure

```
my-app/
├── 📁 app/                    # Next.js 13+ App Router
│   ├── 📄 layout.tsx         # Root layout component
│   ├── 📄 page.tsx           # Home page
│   ├── 📄 globals.css        # Global styles
│   ├── 📁 dashboard/         # Dashboard pages
│   │   └── 📄 page.tsx       # Dashboard main page
│   └── 📁 login/             # Authentication pages
│       └── 📄 page.tsx       # Login page
├── 📁 components/            # Reusable React components
│   └── 📁 ui/                # UI component library
│       ├── 📄 button.tsx     # Button component
│       ├── 📄 card.tsx       # Card component
│       ├── 📄 input.tsx      # Input component
│       └── 📄 ...            # Other UI components
├── 📁 hooks/                 # Custom React hooks
│   └── 📄 use-mobile.ts      # Mobile device detection hook
├── 📁 lib/                   # Utility functions and configurations
│   └── 📄 utils.ts           # Common utility functions
├── 📁 public/                # Static assets
│   ├── 🖼️ favicon.ico        # Site favicon
│   └── 🖼️ *.svg             # SVG icons and images
├── 📄 package.json           # Project dependencies and scripts
├── 📄 tailwind.config.ts     # Tailwind CSS configuration
├── 📄 tsconfig.json          # TypeScript configuration
├── 📄 next.config.ts         # Next.js configuration
└── 📄 components.json        # shadcn/ui configuration
```

---

## 🖥️ Available Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `dev` | Start development server with Turbopack | `npm run dev` |
| `build` | Build production application | `npm run build` |
| `start` | Start production server | `npm run start` |
| `lint` | Run ESLint for code quality | `npm run lint` |

### Development Workflow

```bash
# Start development with hot reload
npm run dev

# Build and test production build
npm run build && npm run start

# Check code quality
npm run lint
```

---

## 🎨 UI Components

This project uses **shadcn/ui** components built on top of **Radix UI** primitives. All components are:

- ✅ **Fully accessible** with ARIA support
- ✅ **Customizable** with CSS variables
- ✅ **Type-safe** with TypeScript
- ✅ **Responsive** and mobile-friendly

### Available Components

| Component | Description | Import |
|-----------|-------------|--------|
| `Button` | Interactive button with variants | `@/components/ui/button` |
| `Card` | Content container with header/footer | `@/components/ui/card` |
| `Input` | Form input with validation | `@/components/ui/input` |
| `Badge` | Status indicators and labels | `@/components/ui/badge` |
| `Sidebar` | Navigation sidebar component | `@/components/ui/sidebar` |
| `Dropdown` | Dropdown menu component | `@/components/ui/dropdown-menu` |

---

## 📱 Responsive Design

The application is fully responsive and optimized for:

- 🖥️ **Desktop** (1920px+) - Full feature set with sidebar navigation
- 💻 **Laptop** (1024px-1919px) - Adaptive layout with collapsible sidebar
- 📱 **Tablet** (768px-1023px) - Touch-optimized interface
- 📱 **Mobile** (320px-767px) - Mobile-first design with bottom navigation

### Breakpoints

```css
/* Tailwind CSS breakpoints used */
sm: 640px   /* Small devices (phones) */
md: 768px   /* Medium devices (tablets) */
lg: 1024px  /* Large devices (laptops) */
xl: 1280px  /* Extra large devices (desktops) */
2xl: 1536px /* 2X Extra large devices */
```

---

## 🔧 Configuration

### Environment Variables

Create a `.env.local` file in the root directory:

```env
# Application Configuration
NEXT_PUBLIC_APP_NAME="Smart Agriculture Dashboard"
NEXT_PUBLIC_APP_VERSION="1.0.0"

# API Configuration
NEXT_PUBLIC_API_URL="https://api.example.com"
API_SECRET_KEY="your-secret-key"

# Database Configuration
DATABASE_URL="postgresql://user:password@localhost:5432/database"

# Authentication (if using NextAuth.js)
NEXTAUTH_SECRET="your-nextauth-secret"
NEXTAUTH_URL="http://localhost:3000"
```

### Tailwind CSS Customization

Modify `tailwind.config.ts` to customize the design system:

```typescript
import type { Config } from "tailwindcss"

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        // Add your custom colors
        primary: "hsl(var(--primary))",
        secondary: "hsl(var(--secondary))",
      },
    },
  },
  plugins: [],
}
```

---

## 🧪 Testing

### Setting Up Tests

```bash
# Install testing dependencies
npm install --save-dev @testing-library/react @testing-library/jest-dom jest jest-environment-jsdom

# Create test configuration
touch jest.config.js
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Generate coverage report
npm run test:coverage
```

---

## 🚀 Deployment

### Vercel (Recommended)

1. **Connect your repository** to [Vercel](https://vercel.com)
2. **Configure environment variables** in the Vercel dashboard
3. **Deploy automatically** on every push to main branch

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=<your-repo-url>)

### Other Platforms

<details>
<summary>Netlify</summary>

```bash
# Build command
npm run build

# Publish directory
out
```
</details>

<details>
<summary>Docker</summary>

```dockerfile
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build
EXPOSE 3000
CMD ["npm", "start"]
```
</details>

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Process

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Code Style

- Use **Prettier** for code formatting
- Follow **ESLint** rules
- Write **meaningful commit messages**
- Add **tests** for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Next.js Team** - For the amazing React framework
- **Vercel** - For the excellent deployment platform
- **shadcn** - For the beautiful UI component library
- **Tailwind CSS** - For the utility-first CSS framework
- **Radix UI** - For accessible UI primitives

---

<div align="center">
  <p>Made with ❤️ for Hackathon Elevate 2025</p>
  <p>
    <a href="#top">⬆️ Back to Top</a>
  </p>
</div>
