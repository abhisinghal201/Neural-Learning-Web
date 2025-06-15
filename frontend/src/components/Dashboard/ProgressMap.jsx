import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import { Brain, Zap, Lock, CheckCircle, Star, ArrowRight } from 'lucide-react';
import * as d3 from 'd3';
import { api } from '../../utils/api';

const ProgressMap = () => {
  const svgRef = useRef();
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoveredNode, setHoveredNode] = useState(null);
  const [mapDimensions, setMapDimensions] = useState({ width: 800, height: 600 });

  // Fetch progress data
  const { data: progressData, isLoading } = useQuery(
    'learningProgress',
    () => api.get('/learning/progress'),
    { refetchInterval: 30000 }
  );

  // Fetch vault data for connection points
  const { data: vaultData } = useQuery(
    'vaultItems',
    () => api.get('/vault/items'),
    { refetchInterval: 60000 }
  );

  // Neural network structure for the progress map
  const generateNeuralStructure = (progressData) => {
    const phases = [
      { id: 1, name: 'Mathematical Foundations', weeks: 12, color: '#00d4ff', x: 0.2, y: 0.8 },
      { id: 2, name: 'Core Machine Learning', weeks: 12, color: '#0099cc', x: 0.4, y: 0.5 },
      { id: 3, name: 'Advanced Topics', weeks: 12, color: '#7b68ee', x: 0.6, y: 0.3 },
      { id: 4, name: 'Mastery & Innovation', weeks: 12, color: '#ff6b9d', x: 0.8, y: 0.2 }
    ];

    const nodes = [];
    const links = [];

    phases.forEach((phase, phaseIndex) => {
      for (let week = 1; week <= phase.weeks; week++) {
        const nodeId = `phase-${phase.id}-week-${week}`;
        const progress = progressData?.data?.progress?.find(
          p => p.phase === phase.id && p.week === week
        );

        // Calculate position in spiral pattern
        const angle = (week - 1) * (Math.PI / 6) + phaseIndex * (Math.PI / 2);
        const radius = 80 + phaseIndex * 100;
        const x = phase.x * mapDimensions.width + Math.cos(angle) * radius;
        const y = phase.y * mapDimensions.height + Math.sin(angle) * radius;

        const node = {
          id: nodeId,
          phase: phase.id,
          week,
          phaseName: phase.name,
          x,
          y,
          color: phase.color,
          status: progress?.status || 'not_started',
          completion: progress?.completion_percentage || 0,
          timeSpent: progress?.time_spent_minutes || 0,
          masteryScore: progress?.mastery_score || 0,
          lessonCount: progressData?.data?.progress?.filter(
            p => p.phase === phase.id && p.week === week
          ).length || 4
        };

        nodes.push(node);

        // Create connections between consecutive weeks
        if (week > 1) {
          links.push({
            source: `phase-${phase.id}-week-${week - 1}`,
            target: nodeId,
            type: 'sequential'
          });
        }

        // Create connections between phases
        if (phaseIndex > 0 && week === 1) {
          links.push({
            source: `phase-${phaseIndex}-week-${phases[phaseIndex - 1].weeks}`,
            target: nodeId,
            type: 'phase-transition'
          });
        }
      }
    });

    // Add vault unlock connections
    if (vaultData?.data?.items) {
      Object.values(vaultData.data.items).flat().forEach(vaultItem => {
        if (vaultItem.unlocked && vaultItem.unlock_condition?.type === 'lesson_complete') {
          const sourceNodeId = `phase-${vaultItem.unlock_condition.phase}-week-${vaultItem.unlock_condition.week}`;
          const sourceNode = nodes.find(n => n.id === sourceNodeId);
          
          if (sourceNode) {
            links.push({
              source: sourceNodeId,
              target: `vault-${vaultItem.id}`,
              type: 'vault-unlock'
            });

            // Add vault node
            nodes.push({
              id: `vault-${vaultItem.id}`,
              type: 'vault',
              category: vaultItem.category,
              title: vaultItem.title,
              icon: vaultItem.icon,
              x: sourceNode.x + (Math.random() - 0.5) * 100,
              y: sourceNode.y - 50,
              color: vaultItem.category === 'secret_archives' ? '#ffd700' :
                     vaultItem.category === 'controversy_files' ? '#ff4757' : '#ff6b9d'
            });
          }
        }
      });
    }

    return { nodes, links };
  };

  // D3 visualization
  useEffect(() => {
    if (!progressData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, links } = generateNeuralStructure(progressData);

    // Create main group
    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Add gradient definitions
    const defs = svg.append('defs');
    
    // Glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '3')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Draw connections
    const linkGroup = g.append('g').attr('class', 'links');
    
    linkGroup.selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('x1', d => {
        const source = nodes.find(n => n.id === d.source);
        return source ? source.x : 0;
      })
      .attr('y1', d => {
        const source = nodes.find(n => n.id === d.source);
        return source ? source.y : 0;
      })
      .attr('x2', d => {
        const target = nodes.find(n => n.id === d.target);
        return target ? target.x : 0;
      })
      .attr('y2', d => {
        const target = nodes.find(n => n.id === d.target);
        return target ? target.y : 0;
      })
      .attr('stroke', d => {
        switch (d.type) {
          case 'sequential': return '#333';
          case 'phase-transition': return '#555';
          case 'vault-unlock': return '#ffd700';
          default: return '#333';
        }
      })
      .attr('stroke-width', d => d.type === 'vault-unlock' ? 2 : 1)
      .attr('stroke-dasharray', d => d.type === 'vault-unlock' ? '5,5' : 'none')
      .style('opacity', 0.6);

    // Draw nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup.selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .style('cursor', 'pointer')
      .on('mouseover', (event, d) => setHoveredNode(d))
      .on('mouseout', () => setHoveredNode(null))
      .on('click', (event, d) => setSelectedNode(d));

    // Node circles
    nodeElements.append('circle')
      .attr('r', d => {
        if (d.type === 'vault') return 12;
        return 15 + (d.completion / 100) * 10;
      })
      .attr('fill', d => {
        if (d.type === 'vault') return d.color;
        
        switch (d.status) {
          case 'completed': return d.color;
          case 'mastered': return '#ffd700';
          case 'in_progress': return `url(#gradient-${d.phase})`;
          default: return '#333';
        }
      })
      .attr('stroke', d => d.status === 'mastered' ? '#ffd700' : d.color)
      .attr('stroke-width', d => d.status === 'mastered' ? 3 : 2)
      .style('filter', d => d.status !== 'not_started' ? 'url(#glow)' : 'none')
      .style('opacity', d => d.status === 'not_started' ? 0.4 : 1);

    // Progress indicators
    nodeElements.filter(d => d.type !== 'vault' && d.completion > 0)
      .append('circle')
      .attr('r', d => 12 + (d.completion / 100) * 8)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 3)
      .attr('stroke-dasharray', d => {
        const circumference = 2 * Math.PI * (12 + (d.completion / 100) * 8);
        const progress = (d.completion / 100) * circumference;
        return `${progress} ${circumference - progress}`;
      })
      .attr('stroke-linecap', 'round')
      .style('transform', 'rotate(-90deg)')
      .style('transform-origin', 'center');

    // Node icons/labels
    nodeElements.append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', d => d.type === 'vault' ? '12px' : '10px')
      .attr('fill', d => d.status === 'not_started' ? '#666' : '#fff')
      .attr('font-weight', 'bold')
      .text(d => {
        if (d.type === 'vault') return d.icon;
        return d.week;
      });

    // Add phase labels
    const phaseLabels = [
      { phase: 1, name: 'Mathematical Foundations', x: 0.2, y: 0.9 },
      { phase: 2, name: 'Core Machine Learning', x: 0.4, y: 0.6 },
      { phase: 3, name: 'Advanced Topics', x: 0.6, y: 0.4 },
      { phase: 4, name: 'Mastery & Innovation', x: 0.8, y: 0.3 }
    ];

    g.append('g')
      .attr('class', 'phase-labels')
      .selectAll('text')
      .data(phaseLabels)
      .enter()
      .append('text')
      .attr('x', d => d.x * mapDimensions.width)
      .attr('y', d => d.y * mapDimensions.height)
      .attr('text-anchor', 'middle')
      .attr('font-size', '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#888')
      .text(d => d.name);

  }, [progressData, vaultData, mapDimensions]);

  // Update dimensions on resize
  useEffect(() => {
    const updateDimensions = () => {
      const container = svgRef.current?.parentElement;
      if (container) {
        setMapDimensions({
          width: container.clientWidth,
          height: container.clientHeight
        });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed': return <CheckCircle className="w-4 h-4 text-green-400" />;
      case 'mastered': return <Star className="w-4 h-4 text-yellow-400" />;
      case 'in_progress': return <Zap className="w-4 h-4 text-blue-400" />;
      default: return <Lock className="w-4 h-4 text-gray-400" />;
    }
  };

  if (isLoading) {
    return (
      <div className="progress-map-container h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-blue-400">
          <Brain className="w-6 h-6 animate-pulse" />
          <span>Loading neural pathways...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="progress-map-container h-full relative bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-xl overflow-hidden">
      {/* Neural Progress Map */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        className="neural-progress-map"
      />

      {/* Node Details Tooltip */}
      <AnimatePresence>
        {hoveredNode && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 10 }}
            className="absolute top-4 left-4 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-lg p-4 min-w-64 z-10"
          >
            {hoveredNode.type === 'vault' ? (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-xl">{hoveredNode.icon}</span>
                  <span className="font-semibold text-white">{hoveredNode.title}</span>
                </div>
                <div className="text-sm text-gray-300 capitalize">
                  {hoveredNode.category.replace('_', ' ')}
                </div>
              </div>
            ) : (
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-white">
                    Phase {hoveredNode.phase} - Week {hoveredNode.week}
                  </span>
                  {getStatusIcon(hoveredNode.status)}
                </div>
                
                <div className="text-sm text-gray-300 mb-3">
                  {hoveredNode.phaseName}
                </div>
                
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Progress:</span>
                    <span className="text-white">{hoveredNode.completion}%</span>
                  </div>
                  
                  <div className="flex justify-between">
                    <span className="text-gray-400">Time Spent:</span>
                    <span className="text-white">{Math.floor(hoveredNode.timeSpent / 60)}h {hoveredNode.timeSpent % 60}m</span>
                  </div>
                  
                  {hoveredNode.masteryScore > 0 && (
                    <div className="flex justify-between">
                      <span className="text-gray-400">Mastery:</span>
                      <span className="text-white">{Math.round(hoveredNode.masteryScore * 100)}%</span>
                    </div>
                  )}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Map Controls */}
      <div className="absolute top-4 right-4 flex flex-col gap-2">
        <button
          className="p-2 bg-gray-800/80 hover:bg-gray-700/80 border border-gray-600 rounded-lg text-white transition-colors"
          onClick={() => {
            const svg = d3.select(svgRef.current);
            svg.transition().duration(750).call(
              d3.zoom().transform,
              d3.zoomIdentity
            );
          }}
        >
          <Brain className="w-4 h-4" />
        </button>
      </div>

      {/* Progress Stats */}
      <div className="absolute bottom-4 left-4 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-lg p-4">
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span className="text-gray-300">Completed</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
            <span className="text-gray-300">Mastered</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span className="text-gray-300">In Progress</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 bg-gray-400 rounded-full"></div>
            <span className="text-gray-300">Locked</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProgressMap;