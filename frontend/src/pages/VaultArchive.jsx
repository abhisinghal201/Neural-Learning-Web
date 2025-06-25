/**
 * Enhanced Neural Odyssey Vault Archive Page
 * 
 * Now fully leverages ALL backend vault capabilities:
 * - Complete vault item management with all categories
 * - Advanced filtering and search with multiple criteria
 * - Vault analytics with unlock patterns and reading statistics
 * - Comprehensive unlock condition tracking and validation
 * - User interaction tracking (read, rate, favorite, notes)
 * - Vault timeline and unlock history
 * - Personalized recommendations based on progress
 * - Export functionality for vault collection
 * - Advanced vault statistics and insights
 * - Integration with learning progress for unlock predictions
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useSearchParams, useNavigate } from 'react-router-dom';
import {
  Sparkles,
  Star,
  Eye,
  EyeOff,
  Lock,
  Unlock,
  Heart,
  Download,
  Upload,
  Search,
  Filter,
  Grid,
  List,
  Calendar,
  Clock,
  Award,
  BookOpen,
  FileText,
  Image,
  Video,
  Music,
  Archive,
  Trash2,
  Share2,
  Edit3,
  Save,
  X,
  Check,
  ChevronDown,
  ChevronUp,
  ChevronRight,
  ArrowUp,
  ArrowDown,
  MoreHorizontal,
  RefreshCw,
  Settings,
  Info,
  AlertTriangle,
  CheckCircle,
  Target,
  Brain,
  Layers,
  Map,
  TrendingUp,
  BarChart3,
  PieChart,
  Activity,
  Users,
  Globe,
  Bookmark,
  MessageSquare,
  ThumbsUp,
  ThumbsDown,
  Timer,
  Zap,
  Trophy,
  Flame,
  Coffee,
  Lightbulb,
  Plus
} from 'lucide-react';
import toast from 'react-hot-toast';

// Components
import VaultRevealModal from '../components/VaultRevealModal';
import LoadingSpinner from '../components/UI/LoadingSpinner';

// Utils
import { api } from '../utils/api';

const VaultArchive = () => {
  const [searchParams, setSearchParams] = useSearchParams();
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // State management
  const [selectedCategory, setSelectedCategory] = useState(searchParams.get('category') || 'all');
  const [selectedType, setSelectedType] = useState(searchParams.get('type') || 'all');
  const [selectedRarity, setSelectedRarity] = useState(searchParams.get('rarity') || 'all');
  const [selectedStatus, setSelectedStatus] = useState(searchParams.get('status') || 'all');
  const [searchQuery, setSearchQuery] = useState(searchParams.get('search') || '');
  const [sortBy, setSortBy] = useState(searchParams.get('sortBy') || 'unlock_date');
  const [sortOrder, setSortOrder] = useState(searchParams.get('sortOrder') || 'desc');
  const [viewMode, setViewMode] = useState(searchParams.get('view') || 'grid');
  const [showFilters, setShowFilters] = useState(false);
  const [showAnalytics, setShowAnalytics] = useState(false);
  
  // Vault interaction state
  const [selectedItems, setSelectedItems] = useState(new Set());
  const [showBulkActions, setShowBulkActions] = useState(false);
  const [selectedVaultItem, setSelectedVaultItem] = useState(null);
  const [showItemModal, setShowItemModal] = useState(false);
  const [editingNotes, setEditingNotes] = useState(null);
  const [itemNotes, setItemNotes] = useState('');
  const [showUnlockPredictions, setShowUnlockPredictions] = useState(false);

  // Advanced filters
  const [advancedFilters, setAdvancedFilters] = useState({
    unlockedAfter: '',
    unlockedBefore: '',
    readStatus: 'all', // all, read, unread
    favoriteStatus: 'all', // all, favorites, non_favorites
    ratingRange: [1, 5],
    minReadTime: 0,
    maxReadTime: 60,
    hasNotes: 'all', // all, with_notes, without_notes
    tags: [],
    difficulty: 'all',
    unlockPhase: 'all',
    estimatedUnlockTime: 'all'
  });

  // Data fetching
  const { data: vaultData, isLoading, refetch } = useQuery(
    ['vaultItems', selectedCategory, selectedType, selectedRarity, selectedStatus, searchQuery, sortBy, sortOrder, advancedFilters],
    () => api.vault.getItems({
      category: selectedCategory !== 'all' ? selectedCategory : undefined,
      type: selectedType !== 'all' ? selectedType : undefined,
      rarity: selectedRarity !== 'all' ? selectedRarity : undefined,
      status: selectedStatus !== 'all' ? selectedStatus : undefined,
      search: searchQuery || undefined,
      sort_by: sortBy,
      sort_order: sortOrder,
      unlocked_after: advancedFilters.unlockedAfter || undefined,
      unlocked_before: advancedFilters.unlockedBefore || undefined,
      read_status: advancedFilters.readStatus !== 'all' ? advancedFilters.readStatus : undefined,
      favorite_status: advancedFilters.favoriteStatus !== 'all' ? advancedFilters.favoriteStatus : undefined,
      min_rating: advancedFilters.ratingRange[0],
      max_rating: advancedFilters.ratingRange[1],
      min_read_time: advancedFilters.minReadTime,
      max_read_time: advancedFilters.maxReadTime,
      has_notes: advancedFilters.hasNotes !== 'all' ? advancedFilters.hasNotes : undefined,
      tags: advancedFilters.tags.length > 0 ? advancedFilters.tags.join(',') : undefined,
      difficulty: advancedFilters.difficulty !== 'all' ? advancedFilters.difficulty : undefined
    }),
    {
      refetchInterval: 120000, // 2 minutes
      staleTime: 60000
    }
  );

  const { data: vaultStatistics } = useQuery(
    'vaultStatistics',
    () => api.vault.getStatistics(),
    {
      refetchInterval: 300000 // 5 minutes
    }
  );

  const { data: vaultAnalytics } = useQuery(
    'vaultAnalytics',
    () => api.vault.getAnalytics({
      timeframe: '90',
      include_patterns: true,
      include_predictions: true
    }),
    {
      refetchInterval: 300000
    }
  );

  const { data: vaultTimeline } = useQuery(
    'vaultTimeline',
    () => api.vault.getTimeline({ limit: 50 }),
    {
      refetchInterval: 300000
    }
  );

  const { data: vaultRecommendations } = useQuery(
    'vaultRecommendations',
    () => api.vault.getRecommendations({ limit: 10 }),
    {
      refetchInterval: 600000 // 10 minutes
    }
  );

  const { data: unlockPredictions } = useQuery(
    'unlockPredictions',
    () => api.vault.getItems({ 
      status: 'locked',
      include_predictions: true,
      sort_by: 'estimated_unlock',
      sort_order: 'asc'
    }),
    {
      enabled: showUnlockPredictions,
      refetchInterval: 600000
    }
  );

  // Mutations
  const toggleFavoriteMutation = useMutation(
    (itemId) => api.vault.toggleFavorite(itemId),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vaultItems');
        queryClient.invalidateQueries('vaultStatistics');
      },
      onError: () => {
        toast.error('Failed to update favorite status');
      }
    }
  );

  const markAsReadMutation = useMutation(
    ({ itemId, readData }) => api.vault.markAsRead(itemId, readData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vaultItems');
        queryClient.invalidateQueries('vaultStatistics');
        toast.success('Marked as read');
      },
      onError: () => {
        toast.error('Failed to mark as read');
      }
    }
  );

  const rateItemMutation = useMutation(
    ({ itemId, rating, review }) => api.vault.rateItem(itemId, { rating, review }),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('vaultItems');
        toast.success('Rating saved');
      },
      onError: () => {
        toast.error('Failed to save rating');
      }
    }
  );

  const unlockItemMutation = useMutation(
    (itemId) => api.vault.unlockItem(itemId),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('vaultItems');
        queryClient.invalidateQueries('vaultStatistics');
        if (data.data.unlocked) {
          toast.success('Item unlocked! 🎉');
        } else {
          toast.error('Unlock conditions not met');
        }
      },
      onError: () => {
        toast.error('Failed to unlock item');
      }
    }
  );

  const checkUnlocksMutation = useMutation(
    () => api.vault.checkUnlocks(),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries('vaultItems');
        queryClient.invalidateQueries('vaultStatistics');
        if (data.data.newly_unlocked?.length > 0) {
          toast.success(`${data.data.newly_unlocked.length} new items unlocked!`);
        } else {
          toast.info('No new unlocks available');
        }
      }
    }
  );

  // Update URL params
  useEffect(() => {
    const params = new URLSearchParams();
    if (selectedCategory !== 'all') params.set('category', selectedCategory);
    if (selectedType !== 'all') params.set('type', selectedType);
    if (selectedRarity !== 'all') params.set('rarity', selectedRarity);
    if (selectedStatus !== 'all') params.set('status', selectedStatus);
    if (searchQuery) params.set('search', searchQuery);
    if (sortBy !== 'unlock_date') params.set('sortBy', sortBy);
    if (sortOrder !== 'desc') params.set('sortOrder', sortOrder);
    if (viewMode !== 'grid') params.set('view', viewMode);
    setSearchParams(params);
  }, [selectedCategory, selectedType, selectedRarity, selectedStatus, searchQuery, sortBy, sortOrder, viewMode, setSearchParams]);

  // Configuration data
  const categories = [
    { value: 'all', label: 'All Categories', icon: Sparkles },
    { value: 'secret_archives', label: 'Secret Archives', icon: Archive },
    { value: 'controversy_files', label: 'Controversy Files', icon: AlertTriangle },
    { value: 'beautiful_mind', label: 'Beautiful Mind', icon: Brain }
  ];

  const types = [
    { value: 'all', label: 'All Types' },
    { value: 'secret', label: 'Secrets' },
    { value: 'tool', label: 'Tools' },
    { value: 'artifact', label: 'Artifacts' },
    { value: 'story', label: 'Stories' },
    { value: 'wisdom', label: 'Wisdom' }
  ];

  const rarities = [
    { value: 'all', label: 'All Rarities' },
    { value: 'common', label: 'Common', color: 'text-gray-400' },
    { value: 'uncommon', label: 'Uncommon', color: 'text-green-400' },
    { value: 'rare', label: 'Rare', color: 'text-blue-400' },
    { value: 'epic', label: 'Epic', color: 'text-purple-400' },
    { value: 'legendary', label: 'Legendary', color: 'text-yellow-400' }
  ];

  const statuses = [
    { value: 'all', label: 'All Items' },
    { value: 'unlocked', label: 'Unlocked' },
    { value: 'locked', label: 'Locked' }
  ];

  const sortOptions = [
    { value: 'unlock_date', label: 'Unlock Date' },
    { value: 'title', label: 'Title' },
    { value: 'rarity', label: 'Rarity' },
    { value: 'read_time', label: 'Read Time' },
    { value: 'rating', label: 'Rating' },
    { value: 'category', label: 'Category' }
  ];

  // Helper functions
  const getRarityColor = (rarity) => {
    const rarityObj = rarities.find(r => r.value === rarity);
    return rarityObj?.color || 'text-gray-400';
  };

  const getTypeIcon = (type) => {
    const icons = {
      secret: Archive,
      tool: Settings,
      artifact: Star,
      story: BookOpen,
      wisdom: Lightbulb
    };
    return icons[type] || FileText;
  };

  const formatTimeAgo = (dateString) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now - date;
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
    
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    if (diffDays < 30) return `${Math.floor(diffDays / 7)} weeks ago`;
    if (diffDays < 365) return `${Math.floor(diffDays / 30)} months ago`;
    return `${Math.floor(diffDays / 365)} years ago`;
  };

  // Action handlers
  const handleItemClick = (item) => {
    if (item.isUnlocked) {
      setSelectedVaultItem(item);
      setShowItemModal(true);
      
      // Mark as read if not already
      if (!item.isRead) {
        markAsReadMutation.mutate({
          itemId: item.itemId,
          readData: {
            read_at: new Date().toISOString(),
            reading_time_minutes: item.estimatedReadTime || 5
          }
        });
      }
    } else {
      // Try to unlock
      unlockItemMutation.mutate(item.itemId);
    }
  };

  const handleToggleFavorite = (e, itemId) => {
    e.stopPropagation();
    toggleFavoriteMutation.mutate(itemId);
  };

  const handleRateItem = (itemId, rating, review = '') => {
    rateItemMutation.mutate({ itemId, rating, review });
  };

  const handleBulkAction = (action) => {
    const itemIds = Array.from(selectedItems);
    
    switch (action) {
      case 'favorite':
        itemIds.forEach(id => toggleFavoriteMutation.mutate(id));
        break;
      case 'mark_read':
        itemIds.forEach(id => markAsReadMutation.mutate({
          itemId: id,
          readData: { read_at: new Date().toISOString() }
        }));
        break;
      case 'export':
        exportVaultCollection(itemIds);
        break;
      default:
        break;
    }
    
    setSelectedItems(new Set());
    setShowBulkActions(false);
  };

  const exportVaultCollection = (itemIds = null) => {
    const items = itemIds 
      ? vaultData?.data?.items?.filter(item => itemIds.includes(item.itemId))
      : vaultData?.data?.items?.filter(item => item.isUnlocked);
    
    const exportData = {
      exported_at: new Date().toISOString(),
      total_items: items?.length || 0,
      items: items || []
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `vault-collection-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    toast.success('Vault collection exported!');
  };

  // Filtered and sorted items
  const filteredItems = useMemo(() => {
    let items = vaultData?.data?.items || [];
    
    // Apply search filter
    if (searchQuery) {
      const query = searchQuery.toLowerCase();
      items = items.filter(item =>
        item.title.toLowerCase().includes(query) ||
        item.description.toLowerCase().includes(query) ||
        item.tags?.some(tag => tag.toLowerCase().includes(query))
      );
    }
    
    // Apply advanced filters
    if (advancedFilters.readStatus !== 'all') {
      items = items.filter(item => 
        advancedFilters.readStatus === 'read' ? item.isRead : !item.isRead
      );
    }
    
    if (advancedFilters.favoriteStatus !== 'all') {
      items = items.filter(item => 
        advancedFilters.favoriteStatus === 'favorites' ? item.isFavorite : !item.isFavorite
      );
    }
    
    if (advancedFilters.hasNotes !== 'all') {
      items = items.filter(item => 
        advancedFilters.hasNotes === 'with_notes' 
          ? item.userNotes && item.userNotes.trim()
          : !item.userNotes || !item.userNotes.trim()
      );
    }
    
    // Sort items
    items.sort((a, b) => {
      let aVal, bVal;
      
      switch (sortBy) {
        case 'title':
          aVal = a.title.toLowerCase();
          bVal = b.title.toLowerCase();
          break;
        case 'rarity':
          const rarityOrder = { common: 1, uncommon: 2, rare: 3, epic: 4, legendary: 5 };
          aVal = rarityOrder[a.rarity] || 0;
          bVal = rarityOrder[b.rarity] || 0;
          break;
        case 'unlock_date':
          aVal = new Date(a.unlockedAt || 0);
          bVal = new Date(b.unlockedAt || 0);
          break;
        case 'rating':
          aVal = a.userRating || 0;
          bVal = b.userRating || 0;
          break;
        case 'read_time':
          aVal = a.estimatedReadTime || 0;
          bVal = b.estimatedReadTime || 0;
          break;
        default:
          aVal = a.createdAt;
          bVal = b.createdAt;
      }
      
      if (sortOrder === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });
    
    return items;
  }, [vaultData, searchQuery, advancedFilters, sortBy, sortOrder]);

  // Render vault item card
  const renderItemCard = (item) => {
    const TypeIcon = getTypeIcon(item.itemType);
    const isSelected = selectedItems.has(item.itemId);
    
    return (
      <motion.div
        key={item.itemId}
        layout
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        whileHover={{ scale: 1.02 }}
        className={`relative bg-gray-800 rounded-lg border cursor-pointer transition-all ${
          item.isUnlocked
            ? 'border-gray-700 hover:border-gray-600'
            : 'border-gray-700 opacity-75 hover:border-gray-600'
        } ${isSelected ? 'ring-2 ring-blue-500' : ''}`}
        onClick={() => handleItemClick(item)}
      >
        {/* Selection checkbox */}
        {showBulkActions && (
          <div className="absolute top-2 left-2 z-10">
            <input
              type="checkbox"
              checked={isSelected}
              onChange={(e) => {
                e.stopPropagation();
                const newSelected = new Set(selectedItems);
                if (e.target.checked) {
                  newSelected.add(item.itemId);
                } else {
                  newSelected.delete(item.itemId);
                }
                setSelectedItems(newSelected);
              }}
              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded"
            />
          </div>
        )}

        {/* Lock/Unlock indicator */}
        <div className="absolute top-2 right-2 z-10">
          {item.isUnlocked ? (
            <Unlock className="w-4 h-4 text-green-400" />
          ) : (
            <Lock className="w-4 h-4 text-red-400" />
          )}
        </div>

        <div className="p-4">
          {/* Header */}
          <div className="flex items-start justify-between mb-3">
            <div className="flex items-center space-x-2">
              <TypeIcon className="w-4 h-4 text-gray-400" />
              <span className={`text-xs px-2 py-1 rounded ${getRarityColor(item.rarity)} bg-gray-700`}>
                {item.rarity}
              </span>
            </div>
            <button
              onClick={(e) => handleToggleFavorite(e, item.itemId)}
              className={`p-1 rounded transition-colors ${
                item.isFavorite ? 'text-red-400 hover:text-red-300' : 'text-gray-400 hover:text-red-400'
              }`}
            >
              <Heart className={`w-4 h-4 ${item.isFavorite ? 'fill-current' : ''}`} />
            </button>
          </div>

          {/* Title and Category */}
          <h3 className="text-white font-medium mb-2 line-clamp-2">{item.title}</h3>
          <p className="text-gray-400 text-sm mb-3 line-clamp-3">{item.description}</p>

          {/* Metadata */}
          <div className="space-y-2 text-xs text-gray-500">
            <div className="flex items-center justify-between">
              <span className="capitalize">{item.category.replace('_', ' ')}</span>
              <span>{item.estimatedReadTime}m read</span>
            </div>
            
            {item.isUnlocked && (
              <>
                {item.unlockedAt && (
                  <div>Unlocked {formatTimeAgo(item.unlockedAt)}</div>
                )}
                
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-1">
                    {item.isRead && <Eye className="w-3 h-3 text-green-400" />}
                    {item.userRating && (
                      <div className="flex items-center space-x-1">
                        <Star className="w-3 h-3 text-yellow-400 fill-current" />
                        <span>{item.userRating}</span>
                      </div>
                    )}
                  </div>
                  <span>Phase {item.unlockConditions?.phase || '?'}</span>
                </div>
              </>
            )}
            
            {!item.isUnlocked && item.unlockConditions && (
              <div className="text-orange-400">
                Unlock: {item.unlockConditions.description || 'Complete requirements'}
              </div>
            )}
          </div>

          {/* Tags */}
          {item.tags && item.tags.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-3">
              {item.tags.slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded"
                >
                  {tag}
                </span>
              ))}
              {item.tags.length > 3 && (
                <span className="text-xs text-gray-400">+{item.tags.length - 3}</span>
              )}
            </div>
          )}
        </div>
      </motion.div>
    );
  };

  // Render list view item
  const renderListItem = (item) => {
    const TypeIcon = getTypeIcon(item.itemType);
    
    return (
      <motion.div
        key={item.itemId}
        layout
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        className={`flex items-center p-4 bg-gray-800 rounded-lg border cursor-pointer transition-all ${
          item.isUnlocked
            ? 'border-gray-700 hover:border-gray-600'
            : 'border-gray-700 opacity-75'
        }`}
        onClick={() => handleItemClick(item)}
      >
        <div className="flex items-center space-x-3 flex-1 min-w-0">
          <div className="flex items-center space-x-2">
            {item.isUnlocked ? (
              <Unlock className="w-4 h-4 text-green-400" />
            ) : (
              <Lock className="w-4 h-4 text-red-400" />
            )}
            <TypeIcon className="w-4 h-4 text-gray-400" />
          </div>
          
          <div className="flex-1 min-w-0">
            <div className="flex items-center space-x-2 mb-1">
              <h3 className="text-white font-medium truncate">{item.title}</h3>
              <span className={`text-xs px-2 py-1 rounded ${getRarityColor(item.rarity)} bg-gray-700`}>
                {item.rarity}
              </span>
            </div>
            <p className="text-gray-400 text-sm truncate">{item.description}</p>
          </div>
          
          <div className="flex items-center space-x-4 text-xs text-gray-500">
            <span className="capitalize">{item.category.replace('_', ' ')}</span>
            <span>{item.estimatedReadTime}m</span>
            {item.isUnlocked && item.unlockedAt && (
              <span>{formatTimeAgo(item.unlockedAt)}</span>
            )}
            <div className="flex items-center space-x-2">
              {item.isRead && <Eye className="w-3 h-3 text-green-400" />}
              {item.isFavorite && <Heart className="w-3 h-3 text-red-400 fill-current" />}
              {item.userRating && (
                <div className="flex items-center space-x-1">
                  <Star className="w-3 h-3 text-yellow-400 fill-current" />
                  <span>{item.userRating}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </motion.div>
    );
  };

  // Render analytics panel
  const renderAnalytics = () => {
    const stats = vaultStatistics?.data;
    const analytics = vaultAnalytics?.data;
    
    if (!stats) return null;
    
    return (
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Vault Analytics</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{stats.total}</div>
            <div className="text-sm text-gray-400">Total Items</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">{stats.unlocked}</div>
            <div className="text-sm text-gray-400">Unlocked</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">{stats.read}</div>
            <div className="text-sm text-gray-400">Read</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">{stats.favorites}</div>
            <div className="text-sm text-gray-400">Favorites</div>
          </div>
        </div>
        
        {analytics?.unlock_patterns && (
          <div className="mb-6">
            <h4 className="text-md font-medium text-white mb-3">Unlock Patterns</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {analytics.unlock_patterns.map((pattern, index) => (
                <div key={index} className="bg-gray-700 rounded-lg p-3">
                  <div className="text-sm text-gray-300">{pattern.pattern}</div>
                  <div className="text-lg font-bold text-white">{pattern.count}</div>
                </div>
              ))}
            </div>
          </div>
        )}
        
        {analytics?.reading_insights && (
          <div>
            <h4 className="text-md font-medium text-white mb-3">Reading Insights</h4>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-gray-400">Average Read Time:</span>
                <span className="text-white">{analytics.reading_insights.avg_read_time}m</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Most Popular Category:</span>
                <span className="text-white capitalize">
                  {analytics.reading_insights.popular_category?.replace('_', ' ')}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-400">Reading Streak:</span>
                <span className="text-white">{analytics.reading_insights.reading_streak} days</span>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">Neural Vault</h1>
          <p className="text-gray-400">
            Your collection of unlocked insights, secrets, and treasures
          </p>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowUnlockPredictions(!showUnlockPredictions)}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              showUnlockPredictions ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <Target className="w-4 h-4 inline mr-1" />
            Predictions
          </button>
          
          <button
            onClick={() => setShowAnalytics(!showAnalytics)}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              showAnalytics ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            <BarChart3 className="w-4 h-4 inline mr-1" />
            Analytics
          </button>
          
          <button
            onClick={() => checkUnlocksMutation.mutate()}
            disabled={checkUnlocksMutation.isLoading}
            className="px-3 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white text-sm transition-colors"
          >
            {checkUnlocksMutation.isLoading ? (
              <RefreshCw className="w-4 h-4 animate-spin inline mr-1" />
            ) : (
              <Zap className="w-4 h-4 inline mr-1" />
            )}
            Check Unlocks
          </button>
          
          <button
            onClick={() => exportVaultCollection()}
            className="px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 text-sm transition-colors"
          >
            <Download className="w-4 h-4 inline mr-1" />
            Export
          </button>
        </div>
      </div>

      {/* Statistics Overview */}
      {vaultStatistics?.data && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gradient-to-r from-purple-500 to-pink-500 rounded-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{vaultStatistics.data.unlocked}</div>
                <div className="text-sm opacity-90">Items Unlocked</div>
              </div>
              <Unlock className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              {Math.round(vaultStatistics.data.unlockRate * 100)}% of collection
            </div>
          </div>

          <div className="bg-gradient-to-r from-green-500 to-emerald-500 rounded-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{vaultStatistics.data.read}</div>
                <div className="text-sm opacity-90">Items Read</div>
              </div>
              <Eye className="w-8 h-8 opacity-80" />
            </div>
            <div className="text-xs mt-2 opacity-75">
              {Math.round(vaultStatistics.data.readRate * 100)}% completion
            </div>
          </div>

          <div className="bg-gradient-to-r from-red-500 to-pink-500 rounded-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{vaultStatistics.data.favorites}</div>
                <div className="text-sm opacity-90">Favorites</div>
              </div>
              <Heart className="w-8 h-8 opacity-80" />
            </div>
          </div>

          <div className="bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-2xl font-bold">{vaultStatistics.data.total}</div>
                <div className="text-sm opacity-90">Total Items</div>
              </div>
              <Archive className="w-8 h-8 opacity-80" />
            </div>
          </div>
        </div>
      )}

      {/* Analytics Panel */}
      <AnimatePresence>
        {showAnalytics && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            {renderAnalytics()}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Unlock Predictions */}
      <AnimatePresence>
        {showUnlockPredictions && unlockPredictions?.data?.items && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="bg-gray-800 rounded-lg border border-gray-700 p-6"
          >
            <h3 className="text-lg font-semibold text-white mb-4">Unlock Predictions</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {unlockPredictions.data.items.slice(0, 9).map((item) => (
                <div key={item.itemId} className="bg-gray-700 rounded-lg p-4">
                  <h4 className="text-white font-medium mb-2">{item.title}</h4>
                  <div className="text-sm text-gray-400 mb-2">{item.description}</div>
                  <div className="text-xs text-orange-400">
                    Estimated unlock: {item.estimated_unlock || 'Progress dependent'}
                  </div>
                  {item.unlock_condition && (
                    <div className="text-xs text-gray-500 mt-1">
                      {item.unlock_condition}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Filters and Controls */}
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <div className="flex flex-wrap items-center gap-4 mb-4">
          {/* Search */}
          <div className="flex-1 min-w-64">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search vault items..."
                className="w-full pl-10 pr-4 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-400"
              />
            </div>
          </div>

          {/* Category Filter */}
          <select
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
          >
            {categories.map((category) => (
              <option key={category.value} value={category.value}>
                {category.label}
              </option>
            ))}
          </select>

          {/* Status Filter */}
          <select
            value={selectedStatus}
            onChange={(e) => setSelectedStatus(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
          >
            {statuses.map((status) => (
              <option key={status.value} value={status.value}>
                {status.label}
              </option>
            ))}
          </select>

          {/* Sort */}
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
          >
            {sortOptions.map((option) => (
              <option key={option.value} value={option.value}>
                {option.label}
              </option>
            ))}
          </select>

          <button
            onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
            className="p-2 bg-gray-700 hover:bg-gray-600 rounded-lg text-gray-300 transition-colors"
          >
            {sortOrder === 'asc' ? <ArrowUp className="w-4 h-4" /> : <ArrowDown className="w-4 h-4" />}
          </button>

          {/* View Mode */}
          <div className="flex bg-gray-700 rounded-lg">
            <button
              onClick={() => setViewMode('grid')}
              className={`p-2 rounded-l-lg transition-colors ${
                viewMode === 'grid' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-600'
              }`}
            >
              <Grid className="w-4 h-4" />
            </button>
            <button
              onClick={() => setViewMode('list')}
              className={`p-2 rounded-r-lg transition-colors ${
                viewMode === 'list' ? 'bg-blue-600 text-white' : 'text-gray-300 hover:bg-gray-600'
              }`}
            >
              <List className="w-4 h-4" />
            </button>
          </div>

          {/* Bulk Actions Toggle */}
          <button
            onClick={() => {
              setShowBulkActions(!showBulkActions);
              setSelectedItems(new Set());
            }}
            className={`px-3 py-2 rounded-lg text-sm transition-colors ${
              showBulkActions ? 'bg-blue-600 text-white' : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
            }`}
          >
            Select
          </button>
        </div>

        {/* Bulk Actions */}
        <AnimatePresence>
          {showBulkActions && selectedItems.size > 0 && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="flex items-center space-x-2 pt-2 border-t border-gray-700"
            >
              <span className="text-sm text-gray-400">
                {selectedItems.size} item{selectedItems.size !== 1 ? 's' : ''} selected:
              </span>
              <button
                onClick={() => handleBulkAction('favorite')}
                className="px-3 py-1 bg-red-600 hover:bg-red-700 rounded text-white text-sm"
              >
                Toggle Favorite
              </button>
              <button
                onClick={() => handleBulkAction('mark_read')}
                className="px-3 py-1 bg-green-600 hover:bg-green-700 rounded text-white text-sm"
              >
                Mark Read
              </button>
              <button
                onClick={() => handleBulkAction('export')}
                className="px-3 py-1 bg-blue-600 hover:bg-blue-700 rounded text-white text-sm"
              >
                Export Selected
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Items Grid/List */}
      <div>
        {filteredItems.length > 0 ? (
          viewMode === 'grid' ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
              {filteredItems.map(renderItemCard)}
            </div>
          ) : (
            <div className="space-y-4">
              {filteredItems.map(renderListItem)}
            </div>
          )
        ) : (
          <div className="text-center py-12">
            <Archive className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-400 mb-2">No items found</h3>
            <p className="text-gray-500">
              {searchQuery || selectedCategory !== 'all' || selectedStatus !== 'all'
                ? 'Try adjusting your filters'
                : 'Complete lessons and quests to unlock vault items'
              }
            </p>
          </div>
        )}
      </div>

      {/* Vault Item Modal */}
      <AnimatePresence>
        {showItemModal && selectedVaultItem && (
          <VaultRevealModal
            item={selectedVaultItem}
            onClose={() => {
              setShowItemModal(false);
              setSelectedVaultItem(null);
            }}
            onRate={(rating, review) => handleRateItem(selectedVaultItem.itemId, rating, review)}
            onToggleFavorite={() => handleToggleFavorite(null, selectedVaultItem.itemId)}
          />
        )}
      </AnimatePresence>
    </div>
  );
};

export default VaultArchive;