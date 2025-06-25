# Neural Odyssey

**Your Personal AI/ML Learning Adventure**

Neural Odyssey is a fully offline, standalone web platform that transforms the journey from software engineer to world‑class AI/ML practitioner into a gamified epic. It combines a first‑principles curriculum with quests, vault treasures, and an in‑browser code playground.

---

## 🚀 Quick Start

### Prerequisites
- **macOS** (tested on macOS 14+ with M4)
- **Node.js** ≥ 18 / **npm**
- **VS Code** (optional but recommended)
- **SQLite3 CLI** (optional; for inspecting `data/user-progress.sqlite`)

### Install & Bootstrap

```bash
# 1. Clone the repo
git clone /path/to/neural-odyssey.git
cd neural-odyssey

# 2. Install root dev dep (concurrently)
npm install

# 3. Bootstrap the SQLite DB
node scripts/init-db.js
