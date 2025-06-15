import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  Vault,
  ArrowLeft,
  Search,
  Filter,
  Eye,
  EyeOff,
  Star,
  Calendar,
  Grid,
  List,
  Lock,
  Unlock,
  Sparkles,
  Award,
  TrendingUp,
  Clock,
  Heart,
  MessageSquare,
  ExternalLink,
  Share2,
  Download,
  BarChart3,
  Lightbulb,
  Flame,
  Brain,
  SortAsc,
  SortDesc
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import VaultRevealModal from '../components/VaultRevealModal';

// Utils
import { api } from '../utils/api';

const VaultArchive = () => {
  const { category } = useParams();
  const navigate = useNavigate();

  // State management
  const [selectedItem, setSelectedItem] = useState(null);
  const [showModal, setShowModal] = useState(false);
  const [viewMode, setViewMode] = useState('grid');
  const [filterCategory, setFilterCategory] = useState(category || 'all');
  const [filterStatus, setFilterStatus] = useState('all');
  const [sortBy, setSortBy] = useState('unlock_date');
  const [sortOrder, setSortOrder] = useState('desc');
  const [searchQuery, setSearchQuery] = useState('');
  const [showUnlockedOnly, setShowUnlockedOnly] = useState(false);

  // Fetch vault data
  const { data: vaultData, isLoading } = useQuery(
    'vaultItems',
    () => api.get('/vault/items'),
    { refetchInterval: 30000 }
  );

  const { data: analyticsData } = useQuery(
    'vaultAnalytics',
    () => api.get('/vault/analytics'),
    { refetchInterval: 60000 }
  );

  const { data: timelineData } = useQuery(
    'vaultTimeline',
    () => api.get('/vault/timeline')
  );

  const { data: upcomingData } = useQuery(
    'upcomingVault',
    () => api.get('/vault/upcoming')
  );

  // Get all vault items as flat array
  const allItems = vaultData?.data?.items ? 
    Object.values(vaultData.data.items).flat() : [];

  // Filter and sort items
  const filteredItems = allItems
    .filter(item => {
      // Category filter
      if (filterCategory !== 'all' && item.category !== filterCategory) return false;
      
      // Status filter
      if (filterStatus === 'unlocked' && !item.unlocked) return false;
      if (filterStatus === 'locked' && item.unlocked) return false;
      if (filterStatus === 'read' && (!item.unlocked || !item.is_read)) return false;
      if (filterStatus === 'unread' && (!item.unlocked || item.is_read)) return false;
      
      // Search query
      if (searchQuery && !item.title.toLowerCase().includes(searchQuery.toLowerCase())) return false;
      
      // Show unlocked only
      if (showUnlockedOnly && !item.unlocked) return false;
      
      return true;
    })
    .sort((a, b) => {
      let aValue, bValue;
      
      switch (sortBy) {
        case 'title':
          aValue = a.title.toLowerCase();
          bValue = b.title.toLowerCase();
          break;
        case 'category':
          aValue = a.category;
          bValue = b.category;
          break;
        case 'unlock_date':
          aValue = a.unlocked_at ? new Date(a.unlocked_at) : new Date(0);
          bValue = b.unlocked_at ? new Date(b.unlocked_at) : new Date(0);
          break;
        case 'rating':
          aValue = a.user_rating || 0;
          bValue = b.user_rating || 0;
          break;
        default:
          return 0;
      }
      
      if (sortOrder === 'asc') {
        return aValue > bValue ? 1 : -1;
      } else {
        return aValue < bValue ? 1 : -1;
      }
    });

  // Category info
  const categoryInfo = {
    all: { 
      name: 'All Categories', 
      icon: '📚', 
      color: 'text-blue-400', 
      bg: 'bg-blue-400/10',
      description: 'Every secret in the Neural Vault'
    },
    secret_archives: { 
      name: 'Secret Archives', 
      icon: '🗝️', 
      color: 'text-yellow-400', 
      bg: 'bg-yellow-400/10',
      description: 'Hidden connections and mind-blowing revelations'
    },
    controversy_files: { 
      name: 'Controversy Files', 
      icon: '⚔️', 
      color: 'text-red-400', 
      bg: 'bg-red-400/10',
      description: 'Drama, feuds, and heated debates in AI history'
    },
    beautiful_mind: { 
      name: 'Beautiful Mind Collection', 
      icon: '💎', 
      color: 'text-purple-400', 
      bg: 'bg-purple-400/10',
      description: 'Mathematical elegance and stunning insights'
    }
  };

  // Handle item click
  const handleItemClick = (item) => {
    if (item.unlocked) {
      setSelectedItem(item);
      setShowModal(true);
    } else {
      toast.error('This vault item is still locked. Complete more lessons to unlock it!');
    }
  };

  // Calculate statistics
  const stats = vaultData?.data?.statistics || {};
  const categoryStats = analyticsData?.data?.by_category || [];

  if (isLoading) {
    return (
      <div className="vault-archive h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-purple-400">
          <Vault className="w-6 h-6 animate-pulse" />
          <span>Loading Neural Vault...</span>
        </div>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="vault-archive h-full overflow-auto"
    >
      <div className="max-w-7xl mx-auto p-6">
        {/* Header */}
        <motion.div
          initial={{ y: -20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          className="flex items-center justify-between mb-6"
        >
          <div className="flex items-center gap-4">
            <button
              onClick={() => navigate('/')}
              className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            
            <div>
              <h1 className="text-3xl font-bold text-white flex items-center gap-3">
                <Vault className="w-8 h-8 text-purple-400" />
                Neural Vault
              </h1>
              <p className="text-gray-400">
                {categoryInfo[filterCategory]?.description || 'Discover the secrets behind machine learning'}
              </p>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="hidden md:flex items-center gap-6 text-sm">
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-400">{stats.unlocked_items || 0}</div>
              <div className="text-gray-400">Unlocked</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-400">{stats.unlock_percentage || 0}%</div>
              <div className="text-gray-400">Complete</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-yellow-400">{analyticsData?.data?.overview?.avg_rating?.toFixed(1) || 'N/A'}</div>
              <div className="text-gray-400">Avg Rating</div>
            </div>
          </div>
        </motion.div>

        {/* Category Navigation */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.1 }}
          className="flex flex-wrap items-center gap-2 mb-6"
        >
          {Object.entries(categoryInfo).map(([key, info]) => (
            <button
              key={key}
              onClick={() => setFilterCategory(key)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                filterCategory === key
                  ? `${info.bg} ${info.color} border border-current`
                  : 'bg-gray-800 text-gray-400 hover:text-white border border-gray-700 hover:border-gray-600'
              }`}
            >
              <span className="text-lg">{info.icon}</span>
              <span className="font-medium">{info.name}</span>
              {key !== 'all' && (
                <span className="bg-current/20 text-current px-2 py-0.5 rounded-full text-xs">
                  {categoryStats.find(c => c.category === key)?.unlocked_count || 0}
                </span>
              )}
            </button>
          ))}
        </motion.div>

        {/* Controls */}
        <motion.div
          initial={{ y: 20, opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="flex flex-wrap items-center justify-between gap-4 mb-6"
        >
          <div className="flex items-center gap-4">
            {/* Search */}
            <div className="relative">
              <Search className="w-4 h-4 absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400" />
              <input
                type="text"
                placeholder="Search vault items..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:border-purple-400 focus:outline-none min-w-[200px]"
              />
            </div>

            {/* Status Filter */}
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value)}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-purple-400 focus:outline-none"
            >
              <option value="all">All Items</option>
              <option value="unlocked">Unlocked</option>
              <option value="locked">Locked</option>
              <option value="read">Read</option>
              <option value="unread">Unread</option>
            </select>

            {/* Sort */}
            <select
              value={`${sortBy}-${sortOrder}`}
              onChange={(e) => {
                const [field, order] = e.target.value.split('-');
                setSortBy(field);
                setSortOrder(order);
              }}
              className="px-3 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white focus:border-purple-400 focus:outline-none"
            >
              <option value="unlock_date-desc">Newest First</option>
              <option value="unlock_date-asc">Oldest First</option>
              <option value="title-asc">Title A-Z</option>
              <option value="title-desc">Title Z-A</option>
              <option value="rating-desc">Highest Rated</option>
              <option value="rating-asc">Lowest Rated</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            {/* Show Unlocked Only Toggle */}
            <button
              onClick={() => setShowUnlockedOnly(!showUnlockedOnly)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                showUnlockedOnly 
                  ? 'bg-purple-500 text-white' 
                  : 'bg-gray-700 text-gray-300 hover:text-white'
              }`}
            >
              {showUnlockedOnly ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
              <span className="text-sm">Unlocked Only</span>
            </button>

            {/* View Mode Toggle */}
            <div className="flex items-center gap-1 bg-gray-800 p-1 rounded-lg">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'grid' ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                <Grid className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-2 rounded transition-colors ${
                  viewMode === 'list' ? 'bg-purple-500 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                <List className="w-4 h-4" />
              </button>
            </div>
          </div>
        </motion.div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Sidebar - Analytics */}
          <motion.div
            initial={{ x: -20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="lg:col-span-1 space-y-6"
          >
            {/* Progress Overview */}
            <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
              <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-purple-400" />
                Discovery Progress
              </h3>
              
              <div className="space-y-3">
                {Object.entries(categoryInfo).slice(1).map(([key, info]) => {
                  const categoryData = categoryStats.find(c => c.category === key);
                  const unlocked = categoryData?.unlocked_count || 0;
                  const total = key === 'secret_archives' ? 3 : 
                               key === 'controversy_files' ? 3 : 3; // Based on our vault-items.json
                  const percentage = total > 0 ? (unlocked / total) * 100 : 0;
                  
                  return (
                    <div key={key}>
                      <div className="flex items-center justify-between mb-1">
                        <span className="text-sm text-gray-300 flex items-center gap-1">
                          <span>{info.icon}</span>
                          {info.name.split(' ')[0]}
                        </span>
                        <span className="text-xs text-gray-400">{unlocked}/{total}</span>
                      </div>
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div 
                          className="h-2 rounded-full transition-all duration-300"
                          style={{ 
                            width: `${percentage}%`,
                            background: `linear-gradient(90deg, ${info.color.replace('text-', '')}, ${info.color.replace('text-', '')}aa)`
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Recent Unlocks */}
            {timelineData?.data?.timeline && timelineData.data.timeline.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-blue-400" />
                  Recent Discoveries
                </h3>
                
                <div className="space-y-3">
                  {timelineData.data.timeline.slice(0, 5).map((item, index) => (
                    <div key={index} className="flex items-center gap-3">
                      <span className="text-lg">{item.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white truncate">{item.title}</div>
                        <div className="text-xs text-gray-400">
                          {new Date(item.unlocked_at).toLocaleDateString()}
                        </div>
                      </div>
                      {item.is_read && <Eye className="w-3 h-3 text-green-400" />}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upcoming Unlocks */}
            {upcomingData?.data?.upcoming_rewards && upcomingData.data.upcoming_rewards.length > 0 && (
              <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4">
                <h3 className="font-semibold text-white mb-4 flex items-center gap-2">
                  <Sparkles className="w-4 h-4 text-yellow-400" />
                  Coming Soon
                </h3>
                
                <div className="space-y-3">
                  {upcomingData.data.upcoming_rewards.slice(0, 3).map((item, index) => (
                    <div key={index} className="flex items-center gap-3 opacity-60">
                      <span className="text-lg">{item.icon}</span>
                      <div className="flex-1 min-w-0">
                        <div className="text-sm text-white truncate">{item.title}</div>
                        <div className="text-xs text-gray-400">{item.estimated_unlock}</div>
                      </div>
                      <Lock className="w-3 h-3 text-gray-400" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>

          {/* Main Content - Vault Items */}
          <motion.div
            initial={{ x: 20, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="lg:col-span-3"
          >
            {filteredItems.length === 0 ? (
              <div className="text-center py-12">
                <Vault className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-semibold text-gray-400 mb-2">No vault items found</h3>
                <p className="text-gray-500">Try adjusting your filters or search query.</p>
              </div>
            ) : (
              <div className={`grid gap-4 ${
                viewMode === 'grid' 
                  ? 'grid-cols-1 md:grid-cols-2 xl:grid-cols-3' 
                  : 'grid-cols-1'
              }`}>
                {filteredItems.map((item, index) => {
                  const categoryData = categoryInfo[item.category];
                  
                  return (
                    <motion.div
                      key={item.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                      onClick={() => handleItemClick(item)}
                      className={`bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-4 transition-all duration-300 cursor-pointer ${
                        item.unlocked 
                          ? 'hover:border-gray-600 hover:bg-gray-800/70' 
                          : 'opacity-60'
                      } ${viewMode === 'list' ? 'flex items-center gap-4' : ''}`}
                    >
                      {/* Icon & Lock Status */}
                      <div className={`relative ${viewMode === 'list' ? 'flex-shrink-0' : 'mb-3'}`}>
                        <div className={`text-4xl ${viewMode === 'list' ? 'text-3xl' : ''}`}>
                          {item.unlocked ? item.icon : '🔒'}
                        </div>
                        {item.unlocked && (
                          <div className="absolute -top-1 -right-1">
                            <Unlock className="w-4 h-4 text-green-400" />
                          </div>
                        )}
                      </div>

                      <div className={`flex-1 ${viewMode === 'list' ? '' : ''}`}>
                        {/* Category Badge */}
                        <div className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium mb-2 ${categoryData.bg} ${categoryData.color}`}>
                          <span>{categoryData.icon}</span>
                          {categoryData.name}
                        </div>

                        {/* Title */}
                        <h3 className={`font-semibold text-white mb-2 ${
                          viewMode === 'list' ? 'text-lg' : ''
                        }`}>
                          {item.unlocked ? item.title : 'Locked Vault Item'}
                        </h3>

                        {/* Status & Metadata */}
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-sm">
                            {item.unlocked && (
                              <>
                                {item.is_read && (
                                  <span className="flex items-center gap-1 text-green-400">
                                    <Eye className="w-3 h-3" />
                                    Read
                                  </span>
                                )}
                                {item.user_rating && (
                                  <span className="flex items-center gap-1 text-yellow-400">
                                    <Star className="w-3 h-3 fill-current" />
                                    {item.user_rating}
                                  </span>
                                )}
                                {item.unlocked_at && (
                                  <span className="text-gray-400">
                                    {new Date(item.unlocked_at).toLocaleDateString()}
                                  </span>
                                )}
                              </>
                            )}
                          </div>

                          {!item.unlocked && (
                            <div className="text-xs text-gray-500">
                              Complete more lessons to unlock
                            </div>
                          )}
                        </div>

                        {/* User Notes Preview */}
                        {item.user_notes && viewMode === 'list' && (
                          <p className="text-sm text-gray-400 mt-2 truncate">
                            "{item.user_notes}"
                          </p>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </div>
            )}
          </motion.div>
        </div>
      </div>

      {/* Vault Reveal Modal */}
      <VaultRevealModal
        vaultItem={selectedItem}
        isOpen={showModal}
        onClose={() => {
          setShowModal(false);
          setSelectedItem(null);
        }}
      />
    </motion.div>
  );
};

export default VaultArchive;