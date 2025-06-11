import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import { 
  Brain, 
  Code, 
  BookOpen, 
  Wrench, 
  Palette, 
  Trophy,
  Star,
  Zap,
  Target,
  Award,
  TrendingUp
} from 'lucide-react';
import * as d3 from 'd3';
import { api } from '../../utils/api';

const SkillTree = () => {
  const svgRef = useRef();
  const [selectedSkill, setSelectedSkill] = useState(null);
  const [hoveredSkill, setHoveredSkill] = useState(null);
  const [treeData, setTreeData] = useState(null);

  // Fetch skill points data
  const { data: skillData, isLoading } = useQuery(
    'skillPoints',
    () => api.get('/learning/analytics'),
    { refetchInterval: 30000 }
  );

  // Skill categories with their properties
  const skillCategories = {
    mathematics: {
      name: 'Mathematics',
      icon: Brain,
      color: '#00d4ff',
      description: 'Linear algebra, calculus, statistics, and mathematical foundations',
      branches: ['Linear Algebra', 'Calculus', 'Statistics', 'Optimization', 'Probability']
    },
    programming: {
      name: 'Programming',
      icon: Code,
      color: '#0099cc',
      description: 'Coding skills, algorithms, and software engineering practices',
      branches: ['Python Mastery', 'Algorithm Design', 'Data Structures', 'Code Quality', 'Debugging']
    },
    theory: {
      name: 'ML Theory',
      icon: BookOpen,
      color: '#7b68ee',
      description: 'Machine learning theory, concepts, and mathematical understanding',
      branches: ['Supervised Learning', 'Unsupervised Learning', 'Deep Learning', 'Reinforcement Learning', 'Ethics']
    },
    applications: {
      name: 'Applications',
      icon: Wrench,
      color: '#ff6b9d',
      description: 'Practical implementation and real-world problem solving',
      branches: ['Computer Vision', 'NLP', 'Time Series', 'Recommendation Systems', 'MLOps']
    },
    creativity: {
      name: 'Creativity',
      icon: Palette,
      color: '#ffa502',
      description: 'Innovation, visualization, and creative problem solving',
      branches: ['Data Visualization', 'Feature Engineering', 'Model Innovation', 'Research Skills', 'Communication']
    },
    persistence: {
      name: 'Persistence',
      icon: Trophy,
      color: '#ff4757',
      description: 'Dedication, consistency, and learning habits',
      branches: ['Daily Practice', 'Streak Building', 'Challenge Completion', 'Growth Mindset', 'Resilience']
    }
  };

  // Generate skill tree structure
  const generateSkillTree = (skillData) => {
    const centerX = 400;
    const centerY = 300;
    const radius = 180;
    
    const skills = skillData?.data?.skills || [];
    const nodes = [];
    const links = [];

    // Create center node
    nodes.push({
      id: 'center',
      type: 'center',
      name: 'Neural Core',
      x: centerX,
      y: centerY,
      level: 0,
      totalPoints: skills.reduce((sum, skill) => sum + skill.total_points, 0)
    });

    // Create skill category nodes in circle around center
    Object.entries(skillCategories).forEach(([key, category], index) => {
      const angle = (index / Object.keys(skillCategories).length) * 2 * Math.PI;
      const x = centerX + Math.cos(angle) * radius;
      const y = centerY + Math.sin(angle) * radius;
      
      const skillInfo = skills.find(s => s.category === key) || { total_points: 0, achievements: 0 };
      const level = Math.floor(skillInfo.total_points / 100) + 1; // Level every 100 points
      const progress = (skillInfo.total_points % 100) / 100;

      const skillNode = {
        id: key,
        type: 'skill',
        name: category.name,
        icon: category.icon,
        color: category.color,
        description: category.description,
        x,
        y,
        level,
        progress,
        points: skillInfo.total_points,
        achievements: skillInfo.achievements,
        branches: category.branches
      };

      nodes.push(skillNode);

      // Link to center
      links.push({
        source: 'center',
        target: key,
        type: 'primary'
      });

      // Create branch nodes for each skill
      category.branches.forEach((branch, branchIndex) => {
        const branchAngle = angle + (branchIndex - 2) * 0.3; // Spread branches around skill
        const branchRadius = 120;
        const branchX = x + Math.cos(branchAngle) * branchRadius;
        const branchY = y + Math.sin(branchAngle) * branchRadius;
        
        // Calculate branch progress (mock data for now)
        const branchProgress = Math.min(1, (skillInfo.total_points / 50) * Math.random());
        const branchLevel = Math.floor(branchProgress * 5) + 1;

        const branchNode = {
          id: `${key}-${branchIndex}`,
          type: 'branch',
          name: branch,
          parentSkill: key,
          color: category.color,
          x: branchX,
          y: branchY,
          level: branchLevel,
          progress: branchProgress,
          unlocked: branchProgress > 0
        };

        nodes.push(branchNode);

        // Link to parent skill
        links.push({
          source: key,
          target: `${key}-${branchIndex}`,
          type: 'branch'
        });
      });
    });

    // Add interconnections between related skills
    const connections = [
      { from: 'mathematics', to: 'theory', strength: 0.8 },
      { from: 'programming', to: 'applications', strength: 0.9 },
      { from: 'theory', to: 'applications', strength: 0.7 },
      { from: 'creativity', to: 'applications', strength: 0.6 },
      { from: 'persistence', to: 'mathematics', strength: 0.5 },
      { from: 'persistence', to: 'programming', strength: 0.5 }
    ];

    connections.forEach(conn => {
      links.push({
        source: conn.from,
        target: conn.to,
        type: 'connection',
        strength: conn.strength
      });
    });

    return { nodes, links };
  };

  // D3 visualization
  useEffect(() => {
    if (!skillData || !svgRef.current) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const { nodes, links } = generateSkillTree(skillData);
    setTreeData({ nodes, links });

    const width = 800;
    const height = 600;

    // Create main group
    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 2])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Add definitions for gradients and filters
    const defs = svg.append('defs');

    // Glow filter
    const filter = defs.append('filter')
      .attr('id', 'glow')
      .attr('x', '-50%')
      .attr('y', '-50%')
      .attr('width', '200%')
      .attr('height', '200%');

    filter.append('feGaussianBlur')
      .attr('stdDeviation', '4')
      .attr('result', 'coloredBlur');

    const feMerge = filter.append('feMerge');
    feMerge.append('feMergeNode').attr('in', 'coloredBlur');
    feMerge.append('feMergeNode').attr('in', 'SourceGraphic');

    // Create gradients for each skill
    Object.entries(skillCategories).forEach(([key, category]) => {
      const gradient = defs.append('radialGradient')
        .attr('id', `gradient-${key}`)
        .attr('cx', '50%')
        .attr('cy', '50%')
        .attr('r', '50%');

      gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', category.color)
        .attr('stop-opacity', 0.8);

      gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', category.color)
        .attr('stop-opacity', 0.3);
    });

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
          case 'primary': return '#555';
          case 'branch': return '#333';
          case 'connection': return '#00d4ff';
          default: return '#333';
        }
      })
      .attr('stroke-width', d => {
        switch (d.type) {
          case 'primary': return 3;
          case 'connection': return 2;
          default: return 1;
        }
      })
      .attr('stroke-dasharray', d => d.type === 'connection' ? '5,5' : 'none')
      .style('opacity', d => {
        switch (d.type) {
          case 'connection': return d.strength * 0.6;
          default: return 0.6;
        }
      });

    // Draw nodes
    const nodeGroup = g.append('g').attr('class', 'nodes');
    
    const nodeElements = nodeGroup.selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('transform', d => `translate(${d.x}, ${d.y})`)
      .style('cursor', 'pointer')
      .on('mouseover', (event, d) => setHoveredSkill(d))
      .on('mouseout', () => setHoveredSkill(null))
      .on('click', (event, d) => setSelectedSkill(d));

    // Node backgrounds (larger circles for glow effect)
    nodeElements.filter(d => d.type !== 'center')
      .append('circle')
      .attr('r', d => {
        switch (d.type) {
          case 'skill': return 35 + d.level * 3;
          case 'branch': return d.unlocked ? 20 : 15;
          default: return 25;
        }
      })
      .attr('fill', d => d.type === 'skill' ? `url(#gradient-${d.id})` : d.color)
      .style('opacity', d => d.type === 'branch' && !d.unlocked ? 0.3 : 0.8)
      .style('filter', 'url(#glow)');

    // Node circles
    nodeElements.append('circle')
      .attr('r', d => {
        switch (d.type) {
          case 'center': return 40;
          case 'skill': return 30 + d.level * 2;
          case 'branch': return d.unlocked ? 18 : 12;
          default: return 20;
        }
      })
      .attr('fill', d => {
        if (d.type === 'center') return 'url(#gradient-center)';
        if (d.type === 'branch') return d.unlocked ? d.color : '#333';
        return d.color;
      })
      .attr('stroke', d => d.type === 'skill' ? '#fff' : 'none')
      .attr('stroke-width', d => d.type === 'skill' ? 2 : 0)
      .style('opacity', d => d.type === 'branch' && !d.unlocked ? 0.5 : 1);

    // Progress rings for skills
    nodeElements.filter(d => d.type === 'skill')
      .append('circle')
      .attr('r', d => 28 + d.level * 2)
      .attr('fill', 'none')
      .attr('stroke', d => d.color)
      .attr('stroke-width', 4)
      .attr('stroke-dasharray', d => {
        const circumference = 2 * Math.PI * (28 + d.level * 2);
        const progress = d.progress * circumference;
        return `${progress} ${circumference - progress}`;
      })
      .attr('stroke-linecap', 'round')
      .style('transform', 'rotate(-90deg)')
      .style('transform-origin', 'center');

    // Level indicators
    nodeElements.filter(d => d.type === 'skill' || d.type === 'center')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', d => d.type === 'center' ? '16px' : '14px')
      .attr('font-weight', 'bold')
      .attr('fill', '#fff')
      .text(d => d.type === 'center' ? '🧠' : d.level);

    // Branch icons/text
    nodeElements.filter(d => d.type === 'branch')
      .append('text')
      .attr('text-anchor', 'middle')
      .attr('dy', '0.35em')
      .attr('font-size', '10px')
      .attr('font-weight', 'bold')
      .attr('fill', d => d.unlocked ? '#fff' : '#666')
      .text(d => d.unlocked ? '✓' : '○');

  }, [skillData]);

  const getSkillIcon = (iconComponent) => {
    const IconComponent = iconComponent;
    return <IconComponent className="w-5 h-5" />;
  };

  const calculateNextLevelProgress = (points) => {
    const currentLevel = Math.floor(points / 100) + 1;
    const pointsInCurrentLevel = points % 100;
    const pointsNeeded = 100 - pointsInCurrentLevel;
    return { currentLevel, pointsInCurrentLevel, pointsNeeded };
  };

  if (isLoading) {
    return (
      <div className="skill-tree-container h-full flex items-center justify-center">
        <div className="flex items-center gap-3 text-blue-400">
          <Star className="w-6 h-6 animate-spin" />
          <span>Loading skill tree...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="skill-tree-container h-full relative bg-gradient-to-br from-purple-900/20 via-blue-900/20 to-gray-900 rounded-xl overflow-hidden">
      {/* Skill Tree Visualization */}
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        className="skill-tree-svg"
        viewBox="0 0 800 600"
      />

      {/* Skill Details Panel */}
      <AnimatePresence>
        {hoveredSkill && (
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            className="absolute top-4 left-4 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-lg p-4 min-w-80 z-10"
          >
            {hoveredSkill.type === 'skill' ? (
              <div>
                <div className="flex items-center gap-3 mb-3">
                  {getSkillIcon(hoveredSkill.icon)}
                  <div>
                    <h3 className="font-bold text-white text-lg">{hoveredSkill.name}</h3>
                    <p className="text-sm text-gray-300">Level {hoveredSkill.level}</p>
                  </div>
                </div>
                
                <p className="text-sm text-gray-300 mb-4">{hoveredSkill.description}</p>
                
                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Total Points:</span>
                    <span className="text-white font-semibold">{hoveredSkill.points}</span>
                  </div>
                  
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400">Achievements:</span>
                    <span className="text-white font-semibold">{hoveredSkill.achievements}</span>
                  </div>
                  
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-400">Next Level Progress</span>
                      <span className="text-white">{Math.round(hoveredSkill.progress * 100)}%</span>
                    </div>
                    <div className="w-full bg-gray-700 rounded-full h-2">
                      <div 
                        className="h-2 rounded-full transition-all duration-300"
                        style={{ 
                          width: `${hoveredSkill.progress * 100}%`,
                          background: `linear-gradient(90deg, ${hoveredSkill.color}, ${hoveredSkill.color}dd)`
                        }}
                      />
                    </div>
                  </div>
                  
                  <div>
                    <h4 className="text-sm font-semibold text-white mb-2">Skill Branches:</h4>
                    <div className="grid grid-cols-2 gap-1 text-xs">
                      {hoveredSkill.branches.map((branch, index) => (
                        <div key={index} className="text-gray-400">
                          • {branch}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ) : hoveredSkill.type === 'branch' ? (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <div 
                    className="w-3 h-3 rounded-full"
                    style={{ backgroundColor: hoveredSkill.color }}
                  />
                  <h3 className="font-semibold text-white">{hoveredSkill.name}</h3>
                </div>
                
                <div className="text-sm text-gray-300 mb-2">
                  {skillCategories[hoveredSkill.parentSkill]?.name} Branch
                </div>
                
                {hoveredSkill.unlocked ? (
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Level:</span>
                      <span className="text-white">{hoveredSkill.level}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Progress:</span>
                      <span className="text-white">{Math.round(hoveredSkill.progress * 100)}%</span>
                    </div>
                  </div>
                ) : (
                  <div className="text-gray-400 text-sm">
                    🔒 Unlock by gaining more {skillCategories[hoveredSkill.parentSkill]?.name} points
                  </div>
                )}
              </div>
            ) : (
              <div>
                <div className="flex items-center gap-2 mb-2">
                  <span className="text-2xl">🧠</span>
                  <h3 className="font-bold text-white text-lg">Neural Core</h3>
                </div>
                <p className="text-sm text-gray-300 mb-3">
                  The heart of your learning journey
                </p>
                <div className="flex justify-between">
                  <span className="text-gray-400">Total Points:</span>
                  <span className="text-white font-semibold">{hoveredSkill.totalPoints}</span>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-lg p-4">
        <h4 className="font-semibold text-white mb-3 flex items-center gap-2">
          <Award className="w-4 h-4" />
          Skill Legend
        </h4>
        <div className="space-y-2 text-sm">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-blue-400 rounded-full flex items-center justify-center text-xs font-bold text-white">5</div>
            <span className="text-gray-300">Skill Level</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 border-2 border-blue-400 rounded-full"></div>
            <span className="text-gray-300">Progress Ring</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-400 rounded-full flex items-center justify-center text-xs">✓</div>
            <span className="text-gray-300">Unlocked Branch</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-gray-600 rounded-full flex items-center justify-center text-xs">○</div>
            <span className="text-gray-300">Locked Branch</span>
          </div>
        </div>
      </div>

      {/* Stats Summary */}
      <div className="absolute top-4 right-4 bg-gray-800/95 backdrop-blur-sm border border-gray-600 rounded-lg p-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingUp className="w-4 h-4 text-blue-400" />
          <span className="font-semibold text-white">Skill Overview</span>
        </div>
        
        <div className="space-y-2 text-sm">
          {skillData?.data?.skills?.slice(0, 3).map((skill, index) => (
            <div key={index} className="flex justify-between items-center">
              <span className="text-gray-300 capitalize">{skill.category}:</span>
              <span className="text-white font-semibold">{skill.total_points}pts</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default SkillTree;