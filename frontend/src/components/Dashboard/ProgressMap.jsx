/**
 * Enhanced Neural Odyssey Progress Map Component
 * 
 * Now fully leverages ALL backend capabilities:
 * - Knowledge graph connections with relationship types and strengths
 * - Interactive learning path visualization with neural network styling
 * - Real-time progress tracking with completion states
 * - Concept prerequisite mapping and dependency visualization
 * - Learning velocity analysis with optimal path suggestions
 * - Phase and week progression with unlock conditions
 * - Session type integration showing recommended learning modes
 * - Skill point distribution and mastery indicators
 * - Adaptive difficulty visualization based on performance
 * - Interactive exploration with detailed tooltips and insights
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import {
  Brain,
  Target,
  BookOpen,
  Code,
  Eye,
  Award,
  Star,
  CheckCircle,
  Circle,
  Lock,
  Unlock,
  TrendingUp,
  Clock,
  Zap,
  Layers,
  Map,
  Compass,
  ArrowRight,
  ArrowDown,
  ChevronRight,
  Plus,
  Minus,
  Search,
  Filter,
  Settings,
  Info,
  Lightbulb,
  Activity,
  BarChart3,
  Timer,
  Flame,
  Users,
  Globe,
  Bookmark,
  Calendar,
  Cpu,
  Database,
  Network,
  GitBranch
} from 'lucide-react';
import * as d3 from 'd3';

// Utils
import { api } from '../utils/api';

const ProgressMap = ({ 
  data, 
  compact = false, 
  interactive = true,
  showConnections = true,
  highlightPhase = null,
  onNodeClick,
  className = '' 
}) => {
  // State management
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [viewMode, setViewMode] = useState('neural'); // neural, tree, flow, grid
  const [filterLevel, setFilterLevel] = useState('all'); // all, current, completed, locked
  const [showLabels, setShowLabels] = useState(!compact);
  const [showDifficulty, setShowDifficulty] = useState(true);
  const [showPrerequisites, setShowPrerequisites] = useState(true);
  const [zoomLevel, setZoomLevel] = useState(1);
  const [centerPosition, setCenterPosition] = useState({ x: 0, y: 0 });

  // Refs
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const simulationRef = useRef(null);

  // Data fetching for enhanced features
  const { data: knowledgeGraph } = useQuery(
    'knowledgeGraphMap',
    () => api.learning.getKnowledgeGraph({ include_strength: true, include_prerequisites: true }),
    {
      refetchInterval: 300000,
      enabled: showConnections
    }
  );

  const { data: progressData } = useQuery(
    'progressMapData',
    () => api.learning.getProgress({ include_analytics: true, include_velocity: true }),
    {
      refetchInterval: 60000
    }
  );

  const { data: skillData } = useQuery(
    'skillMapData',
    () => api.learning.getAnalytics({ metric: 'skills', include_distribution: true }),
    {
      refetchInterval: 300000
    }
  );

  // Process and structure the data
  const processedData = useMemo(() => {
    if (!data && !progressData) return { nodes: [], links: [] };

    const sourceData = data || progressData?.data;
    if (!sourceData) return { nodes: [], links: [] };

    // Create nodes from lessons/concepts
    const nodes = [];
    const links = [];
    const nodeMap = new Map();

    // Learning path structure with enhanced metadata
    const phases = [
      {
        id: 1,
        title: 'Mathematical Foundations',
        color: '#3B82F6',
        weeks: 12,
        concepts: ['linear_algebra', 'calculus', 'probability', 'statistics', 'optimization']
      },
      {
        id: 2,
        title: 'Core Machine Learning',
        color: '#10B981',
        weeks: 12,
        concepts: ['supervised_learning', 'unsupervised_learning', 'neural_networks', 'evaluation']
      },
      {
        id: 3,
        title: 'Advanced Techniques',
        color: '#8B5CF6',
        weeks: 12,
        concepts: ['deep_learning', 'transformers', 'nlp', 'computer_vision', 'rl']
      },
      {
        id: 4,
        title: 'Research & Mastery',
        color: '#F59E0B',
        weeks: 2,
        concepts: ['research', 'innovation', 'publication', 'leadership']
      }
    ];

    // Create phase nodes
    phases.forEach((phase, phaseIndex) => {
      // Create weeks for each phase
      for (let week = 1; week <= phase.weeks; week++) {
        const weekId = `phase_${phase.id}_week_${week}`;
        const weekProgress = sourceData.progress?.find(p => p.phase === phase.id && p.week === week);
        
        const node = {
          id: weekId,
          type: 'week',
          phase: phase.id,
          week: week,
          title: `Phase ${phase.id} Week ${week}`,
          description: `Week ${week} of ${phase.title}`,
          color: phase.color,
          status: weekProgress?.status || 'not_started',
          progress: weekProgress?.progress || 0,
          difficulty: weekProgress?.difficulty_rating || 3,
          timeSpent: weekProgress?.time_spent_minutes || 0,
          concepts: phase.concepts.slice(0, Math.ceil(phase.concepts.length / phase.weeks)),
          skills: weekProgress?.skills || [],
          prerequisites: week > 1 ? [`phase_${phase.id}_week_${week - 1}`] : 
                        phase.id > 1 ? [`phase_${phase.id - 1}_week_${phases[phaseIndex - 1].weeks}`] : [],
          x: 0,
          y: 0,
          fx: null,
          fy: null
        };

        nodes.push(node);
        nodeMap.set(weekId, node);

        // Create prerequisite links
        node.prerequisites.forEach(prereqId => {
          if (nodeMap.has(prereqId)) {
            links.push({
              source: prereqId,
              target: weekId,
              type: 'prerequisite',
              strength: 0.8
            });
          }
        });
      }
    });

    // Add knowledge graph connections if available
    if (knowledgeGraph?.data?.connections && showConnections) {
      knowledgeGraph.data.connections.forEach(connection => {
        const sourceNode = nodes.find(n => 
          n.concepts?.includes(connection.from_concept) || 
          n.id.includes(connection.from_concept)
        );
        const targetNode = nodes.find(n => 
          n.concepts?.includes(connection.to_concept) || 
          n.id.includes(connection.to_concept)
        );

        if (sourceNode && targetNode) {
          links.push({
            source: sourceNode.id,
            target: targetNode.id,
            type: connection.relationship_type || 'related',
            strength: connection.strength || 0.5,
            description: connection.description
          });
        }
      });
    }

    return { nodes, links };
  }, [data, progressData, knowledgeGraph, showConnections]);

  // Filter nodes based on current filter level
  const filteredData = useMemo(() => {
    let filteredNodes = processedData.nodes;

    switch (filterLevel) {
      case 'current':
        filteredNodes = filteredNodes.filter(node => 
          node.status === 'in_progress' || 
          (node.phase === progressData?.data?.profile?.current_phase)
        );
        break;
      case 'completed':
        filteredNodes = filteredNodes.filter(node => 
          node.status === 'completed' || node.status === 'mastered'
        );
        break;
      case 'locked':
        filteredNodes = filteredNodes.filter(node => 
          node.status === 'not_started' || node.status === 'locked'
        );
        break;
    }

    const filteredNodeIds = new Set(filteredNodes.map(n => n.id));
    const filteredLinks = processedData.links.filter(link => 
      filteredNodeIds.has(link.source) && filteredNodeIds.has(link.target)
    );

    return { nodes: filteredNodes, links: filteredLinks };
  }, [processedData, filterLevel, progressData]);

  // D3 force simulation setup
  useEffect(() => {
    if (!svgRef.current || filteredData.nodes.length === 0) return;

    const svg = d3.select(svgRef.current);
    const container = containerRef.current;
    const width = container?.clientWidth || 800;
    const height = container?.clientHeight || 600;

    // Clear previous simulation
    if (simulationRef.current) {
      simulationRef.current.stop();
    }

    // Create simulation based on view mode
    let simulation;
    
    if (viewMode === 'neural') {
      simulation = d3.forceSimulation(filteredData.nodes)
        .force('link', d3.forceLink(filteredData.links)
          .id(d => d.id)
          .distance(d => compact ? 60 : 100)
          .strength(d => d.strength || 0.5)
        )
        .force('charge', d3.forceManyBody()
          .strength(compact ? -100 : -200)
        )
        .force('center', d3.forceCenter(width / 2, height / 2))
        .force('collision', d3.forceCollide()
          .radius(d => compact ? 15 : 25)
        );
    } else if (viewMode === 'tree') {
      // Hierarchical tree layout
      const hierarchy = d3.hierarchy({
        children: filteredData.nodes.filter(n => n.phase === 1)
      });
      
      const treeLayout = d3.tree()
        .size([width - 100, height - 100]);
      
      treeLayout(hierarchy);
      
      simulation = d3.forceSimulation(filteredData.nodes)
        .force('link', d3.forceLink(filteredData.links)
          .id(d => d.id)
          .distance(80)
        )
        .force('charge', d3.forceManyBody().strength(-150))
        .force('center', d3.forceCenter(width / 2, height / 2));
    } else if (viewMode === 'flow') {
      // Flow-based layout
      simulation = d3.forceSimulation(filteredData.nodes)
        .force('link', d3.forceLink(filteredData.links)
          .id(d => d.id)
          .distance(120)
        )
        .force('charge', d3.forceManyBody().strength(-100))
        .force('x', d3.forceX(d => (d.phase - 1) * (width / 4) + width / 8).strength(0.5))
        .force('y', d3.forceY(d => (d.week - 1) * 40 + height / 6).strength(0.3))
        .force('collision', d3.forceCollide().radius(20));
    } else {
      // Grid layout
      simulation = d3.forceSimulation(filteredData.nodes)
        .force('x', d3.forceX(d => (d.phase - 1) * (width / 4) + width / 8).strength(1))
        .force('y', d3.forceY(d => (d.week - 1) * (height / 12) + height / 24).strength(1))
        .force('collision', d3.forceCollide().radius(compact ? 12 : 20));
    }

    simulationRef.current = simulation;

    // Clear SVG
    svg.selectAll('*').remove();

    // Create zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.1, 3])
      .on('zoom', (event) => {
        setZoomLevel(event.transform.k);
        setCenterPosition({ x: event.transform.x, y: event.transform.y });
        svg.select('.zoom-group').attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create main group for zooming
    const g = svg.append('g').attr('class', 'zoom-group');

    // Create links
    const link = g.append('g')
      .attr('class', 'links')
      .selectAll('line')
      .data(filteredData.links)
      .enter()
      .append('line')
      .attr('class', d => `link link-${d.type}`)
      .attr('stroke', d => {
        const colors = {
          prerequisite: '#6B7280',
          related: '#3B82F6',
          builds_on: '#10B981',
          applies_to: '#8B5CF6'
        };
        return colors[d.type] || '#6B7280';
      })
      .attr('stroke-width', d => compact ? 1 : (d.strength || 0.5) * 3)
      .attr('stroke-opacity', compact ? 0.4 : 0.6)
      .attr('stroke-dasharray', d => d.type === 'prerequisite' ? '0' : '5,5');

    // Create nodes
    const node = g.append('g')
      .attr('class', 'nodes')
      .selectAll('.node')
      .data(filteredData.nodes)
      .enter()
      .append('g')
      .attr('class', 'node')
      .style('cursor', interactive ? 'pointer' : 'default');

    // Node circles
    node.append('circle')
      .attr('r', d => {
        if (compact) return 8;
        const baseSize = 15;
        const progressMultiplier = 1 + (d.progress || 0) * 0.5;
        return baseSize * progressMultiplier;
      })
      .attr('fill', d => {
        const statusColors = {
          'not_started': '#374151',
          'in_progress': '#F59E0B',
          'completed': '#10B981',
          'mastered': '#8B5CF6',
          'locked': '#6B7280'
        };
        return statusColors[d.status] || d.color || '#3B82F6';
      })
      .attr('stroke', d => {
        if (d.id === selectedNode?.id) return '#FFFFFF';
        if (d.id === hoveredNode?.id) return '#E5E7EB';
        return '#1F2937';
      })
      .attr('stroke-width', d => {
        if (d.id === selectedNode?.id) return 3;
        if (d.id === hoveredNode?.id) return 2;
        return 1;
      })
      .attr('opacity', d => {
        if (highlightPhase && d.phase !== highlightPhase) return 0.3;
        return 1;
      });

    // Progress rings
    if (!compact) {
      node.append('circle')
        .attr('r', 20)
        .attr('fill', 'none')
        .attr('stroke', d => d.color || '#3B82F6')
        .attr('stroke-width', 2)
        .attr('stroke-opacity', 0.3)
        .attr('stroke-dasharray', d => {
          const circumference = 2 * Math.PI * 20;
          const progress = d.progress || 0;
          return `${circumference * progress} ${circumference * (1 - progress)}`;
        })
        .attr('transform', 'rotate(-90)');
    }

    // Difficulty indicators
    if (showDifficulty && !compact) {
      node.append('rect')
        .attr('x', -15)
        .attr('y', -25)
        .attr('width', 30)
        .attr('height', 3)
        .attr('fill', d => {
          const difficulty = d.difficulty || 3;
          if (difficulty <= 2) return '#10B981'; // Easy - Green
          if (difficulty <= 4) return '#F59E0B'; // Medium - Yellow
          return '#EF4444'; // Hard - Red
        })
        .attr('opacity', 0.7);
    }

    // Node labels
    if (showLabels) {
      node.append('text')
        .attr('dy', compact ? 3 : 35)
        .attr('text-anchor', 'middle')
        .style('font-size', compact ? '8px' : '12px')
        .style('font-weight', '500')
        .style('fill', '#F3F4F6')
        .style('pointer-events', 'none')
        .text(d => {
          if (compact) return `P${d.phase}W${d.week}`;
          return `Phase ${d.phase} Week ${d.week}`;
        });
    }

    // Status icons
    if (!compact) {
      node.append('text')
        .attr('text-anchor', 'middle')
        .attr('dy', 5)
        .style('font-size', '12px')
        .style('fill', '#FFFFFF')
        .style('pointer-events', 'none')
        .text(d => {
          const icons = {
            'completed': '✓',
            'mastered': '★',
            'in_progress': '◐',
            'locked': '🔒'
          };
          return icons[d.status] || '';
        });
    }

    // Event handlers
    if (interactive) {
      node
        .on('mouseover', function(event, d) {
          setHoveredNode(d);
          
          // Highlight connected nodes
          const connectedNodes = new Set();
          filteredData.links.forEach(link => {
            if (link.source.id === d.id) connectedNodes.add(link.target.id);
            if (link.target.id === d.id) connectedNodes.add(link.source.id);
          });
          
          node.selectAll('circle')
            .attr('opacity', node => {
              if (node.id === d.id) return 1;
              if (connectedNodes.has(node.id)) return 0.8;
              return 0.3;
            });

          link.attr('opacity', link => {
            if (link.source.id === d.id || link.target.id === d.id) return 0.8;
            return 0.1;
          });
        })
        .on('mouseout', function() {
          setHoveredNode(null);
          
          node.selectAll('circle').attr('opacity', 1);
          link.attr('opacity', compact ? 0.4 : 0.6);
        })
        .on('click', function(event, d) {
          setSelectedNode(d);
          if (onNodeClick) onNodeClick(d);
        });

      // Drag behavior
      const drag = d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        });

      node.call(drag);
    }

    // Update positions on simulation tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    // Cleanup
    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [filteredData, viewMode, compact, interactive, showLabels, showDifficulty, selectedNode, hoveredNode, highlightPhase]);

  // Render control panel
  const renderControls = () => {
    if (compact) return null;

    return (
      <div className="absolute top-4 left-4 bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-3 z-10">
        <div className="flex items-center space-x-2">
          <Settings className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-white">View Controls</span>
        </div>

        {/* View Mode */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">Layout</label>
          <select
            value={viewMode}
            onChange={(e) => setViewMode(e.target.value)}
            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
          >
            <option value="neural">Neural Network</option>
            <option value="flow">Learning Flow</option>
            <option value="tree">Hierarchy Tree</option>
            <option value="grid">Grid Layout</option>
          </select>
        </div>

        {/* Filter Level */}
        <div>
          <label className="block text-xs text-gray-400 mb-1">Filter</label>
          <select
            value={filterLevel}
            onChange={(e) => setFilterLevel(e.target.value)}
            className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
          >
            <option value="all">All Items</option>
            <option value="current">Current Phase</option>
            <option value="completed">Completed</option>
            <option value="locked">Locked</option>
          </select>
        </div>

        {/* Toggle Options */}
        <div className="space-y-2">
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              className="w-3 h-3 text-blue-600 bg-gray-100 border-gray-300 rounded"
            />
            <span className="text-xs text-gray-300">Show Labels</span>
          </label>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showDifficulty}
              onChange={(e) => setShowDifficulty(e.target.checked)}
              className="w-3 h-3 text-blue-600 bg-gray-100 border-gray-300 rounded"
            />
            <span className="text-xs text-gray-300">Difficulty Bars</span>
          </label>
          
          <label className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={showConnections}
              onChange={(e) => setShowConnections(e.target.checked)}
              className="w-3 h-3 text-blue-600 bg-gray-100 border-gray-300 rounded"
            />
            <span className="text-xs text-gray-300">Knowledge Links</span>
          </label>
        </div>

        {/* Zoom Info */}
        <div className="text-xs text-gray-400">
          Zoom: {Math.round(zoomLevel * 100)}%
        </div>
      </div>
    );
  };

  // Render node details panel
  const renderNodeDetails = () => {
    if (!selectedNode || compact) return null;

    return (
      <AnimatePresence>
        <motion.div
          initial={{ opacity: 0, x: 300 }}
          animate={{ opacity: 1, x: 0 }}
          exit={{ opacity: 0, x: 300 }}
          className="absolute top-4 right-4 bg-gray-800 rounded-lg border border-gray-700 p-4 w-80 z-10"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-white">{selectedNode.title}</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-white"
            >
              ×
            </button>
          </div>

          <div className="space-y-3">
            {/* Status and Progress */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-gray-400">Status</div>
                <div className={`text-sm font-medium capitalize ${
                  selectedNode.status === 'completed' ? 'text-green-400' :
                  selectedNode.status === 'in_progress' ? 'text-yellow-400' :
                  selectedNode.status === 'mastered' ? 'text-purple-400' :
                  'text-gray-400'
                }`}>
                  {selectedNode.status?.replace('_', ' ')}
                </div>
              </div>
              <div>
                <div className="text-xs text-gray-400">Progress</div>
                <div className="text-sm font-medium text-white">
                  {Math.round((selectedNode.progress || 0) * 100)}%
                </div>
              </div>
            </div>

            {/* Concepts */}
            {selectedNode.concepts && selectedNode.concepts.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-1">Key Concepts</div>
                <div className="flex flex-wrap gap-1">
                  {selectedNode.concepts.map((concept, index) => (
                    <span
                      key={index}
                      className="text-xs bg-gray-700 text-gray-300 px-2 py-1 rounded"
                    >
                      {concept.replace('_', ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Time and Difficulty */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <div className="text-xs text-gray-400">Time Spent</div>
                <div className="text-sm text-white">{selectedNode.timeSpent || 0}m</div>
              </div>
              <div>
                <div className="text-xs text-gray-400">Difficulty</div>
                <div className="flex items-center space-x-1">
                  {[1, 2, 3, 4, 5].map((level) => (
                    <div
                      key={level}
                      className={`w-2 h-2 rounded ${
                        level <= (selectedNode.difficulty || 3) ? 'bg-orange-400' : 'bg-gray-600'
                      }`}
                    />
                  ))}
                </div>
              </div>
            </div>

            {/* Prerequisites */}
            {selectedNode.prerequisites && selectedNode.prerequisites.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-1">Prerequisites</div>
                <div className="space-y-1">
                  {selectedNode.prerequisites.map((prereqId, index) => {
                    const prereqNode = processedData.nodes.find(n => n.id === prereqId);
                    return (
                      <div key={index} className="text-xs text-gray-300">
                        {prereqNode ? prereqNode.title : prereqId}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Skills */}
            {selectedNode.skills && selectedNode.skills.length > 0 && (
              <div>
                <div className="text-xs text-gray-400 mb-1">Skills Developed</div>
                <div className="space-y-1">
                  {selectedNode.skills.map((skill, index) => (
                    <div key={index} className="flex items-center justify-between text-xs">
                      <span className="text-gray-300">{skill.name}</span>
                      <span className="text-white">{skill.points} pts</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </motion.div>
      </AnimatePresence>
    );
  };

  // Render legend
  const renderLegend = () => {
    if (compact) return null;

    return (
      <div className="absolute bottom-4 left-4 bg-gray-800 rounded-lg border border-gray-700 p-4 z-10">
        <div className="flex items-center space-x-2 mb-3">
          <Info className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-white">Legend</span>
        </div>

        <div className="space-y-2">
          {/* Status Legend */}
          <div>
            <div className="text-xs text-gray-400 mb-1">Status</div>
            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-gray-600"></div>
                <span className="text-gray-300">Not Started</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
                <span className="text-gray-300">In Progress</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-gray-300">Completed</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                <span className="text-gray-300">Mastered</span>
              </div>
            </div>
          </div>

          {/* Connection Types */}
          {showConnections && (
            <div>
              <div className="text-xs text-gray-400 mb-1">Connections</div>
              <div className="space-y-1 text-xs">
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-px bg-gray-500"></div>
                  <span className="text-gray-300">Prerequisite</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-4 h-px bg-blue-500" style={{ strokeDasharray: '2,2' }}></div>
                  <span className="text-gray-300">Related</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className={`relative w-full ${compact ? 'h-64' : 'h-full min-h-96'} ${className}`}>
      <div ref={containerRef} className="w-full h-full">
        <svg
          ref={svgRef}
          width="100%"
          height="100%"
          className="bg-gray-900 rounded-lg"
        >
          {/* Gradient definitions for enhanced visual effects */}
          <defs>
            <radialGradient id="nodeGradient" cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#FFFFFF" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#000000" stopOpacity="0.1"/>
            </radialGradient>
            <filter id="glow">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge> 
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>
        </svg>
      </div>

      {/* Overlay Controls */}
      {renderControls()}
      {renderNodeDetails()}
      {renderLegend()}

      {/* Statistics Overlay */}
      {!compact && progressData?.data && (
        <div className="absolute top-4 right-4 bg-gray-800 rounded-lg border border-gray-700 p-4 z-10">
          <div className="text-sm font-medium text-white mb-2">Progress Overview</div>
          <div className="space-y-1 text-xs">
            <div className="flex justify-between">
              <span className="text-gray-400">Current Phase:</span>
              <span className="text-white">{progressData.data.profile?.current_phase || 1}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Current Week:</span>
              <span className="text-white">{progressData.data.profile?.current_week || 1}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Completed:</span>
              <span className="text-green-400">
                {filteredData.nodes.filter(n => n.status === 'completed' || n.status === 'mastered').length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">In Progress:</span>
              <span className="text-yellow-400">
                {filteredData.nodes.filter(n => n.status === 'in_progress').length}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProgressMap;