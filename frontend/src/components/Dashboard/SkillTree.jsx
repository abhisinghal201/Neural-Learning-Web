/**
 * Neural Odyssey Skill Tree Component
 *
 * Interactive visualization of the user's skill progression through the ML curriculum.
 * Displays a branching tree structure showing mastery levels across different skill
 * categories, with interconnected nodes representing skill dependencies and achievements.
 *
 * Features:
 * - 6 main skill categories with subcategories
 * - Interactive skill nodes with hover details
 * - Visual progression paths and dependencies
 * - Skill level calculations and badges
 * - Achievement unlocks and milestones
 * - Animated skill point accumulation
 * - Responsive tree layout
 * - Detailed skill information modals
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useQuery } from 'react-query';
import { useNavigate } from 'react-router-dom';
import {
    Brain,
    Code,
    BookOpen,
    Target,
    Lightbulb,
    Zap,
    Star,
    Lock,
    Unlock,
    Trophy,
    Award,
    TrendingUp,
    Circle,
    CheckCircle,
    Plus,
    Minus,
    RotateCcw,
    Eye,
    EyeOff,
    Info,
    ChevronRight,
    Flame,
    Sparkles,
    Crown,
    Shield,
    Gem
} from 'lucide-react';
import { api } from '../../utils/api';

const SkillTree = ({ profile, className = '' }) => {
    const navigate = useNavigate();
    const [selectedSkill, setSelectedSkill] = useState(null);
    const [treeLayout, setTreeLayout] = useState('radial'); // 'radial' | 'hierarchical'
    const [zoom, setZoom] = useState(1);
    const [showConnections, setShowConnections] = useState(true);
    const [hoveredSkill, setHoveredSkill] = useState(null);
    const [skillFilter, setSkillFilter] = useState('all'); // 'all' | 'unlocked' | 'locked'
    const containerRef = useRef(null);

    // Fetch skill points data
    const { data: skillData, isLoading, error, refetch } = useQuery(
        'skillPoints',
        () => api.get('/learning/analytics?timeframe=all'),
        {
            refetchInterval: 60000,
            staleTime: 30000
        }
    );

    // Fetch learning progress for skill calculations
    const { data: progressData } = useQuery(
        'learningProgressForSkills',
        () => api.get('/learning/progress'),
        {
            refetchInterval: 60000
        }
    );

    // Skill tree configuration
    const skillCategories = useMemo(() => [
        {
            id: 'mathematics',
            name: 'Mathematics',
            icon: Brain,
            color: 'from-blue-500 to-blue-600',
            lightColor: 'bg-blue-100',
            position: { x: 0, y: -200 },
            skills: [
                {
                    id: 'linear_algebra',
                    name: 'Linear Algebra',
                    description: 'Vectors, matrices, eigenvalues, and transformations',
                    prerequisites: [],
                    maxLevel: 10,
                    pointsPerLevel: 50,
                    achievements: ['Matrix Master', 'Eigenvalue Expert', 'Transformation Guru']
                },
                {
                    id: 'calculus',
                    name: 'Calculus',
                    description: 'Derivatives, integrals, and optimization',
                    prerequisites: ['linear_algebra'],
                    maxLevel: 10,
                    pointsPerLevel: 50,
                    achievements: ['Derivative Detective', 'Integral Innovator', 'Optimization Oracle']
                },
                {
                    id: 'probability',
                    name: 'Probability',
                    description: 'Random variables, distributions, and Bayesian thinking',
                    prerequisites: ['calculus'],
                    maxLevel: 10,
                    pointsPerLevel: 60,
                    achievements: ['Probability Pioneer', 'Bayesian Brain', 'Distribution Deity']
                },
                {
                    id: 'statistics',
                    name: 'Statistics',
                    description: 'Hypothesis testing, confidence intervals, and inference',
                    prerequisites: ['probability'],
                    maxLevel: 10,
                    pointsPerLevel: 60,
                    achievements: ['Statistics Sage', 'Hypothesis Hero', 'Inference Intellect']
                }
            ]
        },
        {
            id: 'programming',
            name: 'Programming',
            icon: Code,
            color: 'from-green-500 to-green-600',
            lightColor: 'bg-green-100',
            position: { x: 200, y: -100 },
            skills: [
                {
                    id: 'python_basics',
                    name: 'Python Basics',
                    description: 'Syntax, data structures, and control flow',
                    prerequisites: [],
                    maxLevel: 8,
                    pointsPerLevel: 30,
                    achievements: ['Python Padawan', 'Code Craftsman', 'Logic Luminary']
                },
                {
                    id: 'data_manipulation',
                    name: 'Data Manipulation',
                    description: 'Pandas, NumPy, and data preprocessing',
                    prerequisites: ['python_basics'],
                    maxLevel: 10,
                    pointsPerLevel: 40,
                    achievements: ['Data Dancer', 'Pandas Pro', 'NumPy Ninja']
                },
                {
                    id: 'algorithm_implementation',
                    name: 'Algorithm Implementation',
                    description: 'Building ML algorithms from scratch',
                    prerequisites: ['data_manipulation'],
                    maxLevel: 12,
                    pointsPerLevel: 70,
                    achievements: ['Algorithm Architect', 'Code Conjurer', 'Implementation Idol']
                },
                {
                    id: 'optimization',
                    name: 'Optimization',
                    description: 'Gradient descent, hyperparameter tuning, and efficiency',
                    prerequisites: ['algorithm_implementation'],
                    maxLevel: 10,
                    pointsPerLevel: 80,
                    achievements: ['Optimization Oracle', 'Efficiency Expert', 'Performance Prodigy']
                }
            ]
        },
        {
            id: 'theory',
            name: 'Theory',
            icon: BookOpen,
            color: 'from-purple-500 to-purple-600',
            lightColor: 'bg-purple-100',
            position: { x: -200, y: -100 },
            skills: [
                {
                    id: 'ml_fundamentals',
                    name: 'ML Fundamentals',
                    description: 'Supervised, unsupervised, and reinforcement learning',
                    prerequisites: [],
                    maxLevel: 8,
                    pointsPerLevel: 40,
                    achievements: ['Theory Titan', 'Concept Crusader', 'Foundation Fellow']
                },
                {
                    id: 'model_evaluation',
                    name: 'Model Evaluation',
                    description: 'Cross-validation, metrics, and bias-variance tradeoff',
                    prerequisites: ['ml_fundamentals'],
                    maxLevel: 10,
                    pointsPerLevel: 50,
                    achievements: ['Evaluation Expert', 'Metric Master', 'Validation Virtuoso']
                },
                {
                    id: 'deep_learning',
                    name: 'Deep Learning',
                    description: 'Neural networks, backpropagation, and architectures',
                    prerequisites: ['model_evaluation'],
                    maxLevel: 12,
                    pointsPerLevel: 80,
                    achievements: ['Deep Diver', 'Neural Navigator', 'Architecture Ace']
                },
                {
                    id: 'advanced_topics',
                    name: 'Advanced Topics',
                    description: 'Transformers, GANs, and cutting-edge research',
                    prerequisites: ['deep_learning'],
                    maxLevel: 15,
                    pointsPerLevel: 100,
                    achievements: ['Research Rookie', 'Innovation Icon', 'Future Forecaster']
                }
            ]
        },
        {
            id: 'applications',
            name: 'Applications',
            icon: Target,
            color: 'from-orange-500 to-orange-600',
            lightColor: 'bg-orange-100',
            position: { x: 200, y: 100 },
            skills: [
                {
                    id: 'computer_vision',
                    name: 'Computer Vision',
                    description: 'Image processing, CNNs, and visual recognition',
                    prerequisites: ['deep_learning'],
                    maxLevel: 10,
                    pointsPerLevel: 90,
                    achievements: ['Vision Virtuoso', 'Image Interpreter', 'CNN Champion']
                },
                {
                    id: 'nlp',
                    name: 'Natural Language Processing',
                    description: 'Text processing, transformers, and language models',
                    prerequisites: ['deep_learning'],
                    maxLevel: 10,
                    pointsPerLevel: 90,
                    achievements: ['Language Lord', 'Text Tamer', 'NLP Navigator']
                },
                {
                    id: 'time_series',
                    name: 'Time Series Analysis',
                    description: 'Forecasting, LSTM, and temporal patterns',
                    prerequisites: ['model_evaluation'],
                    maxLevel: 8,
                    pointsPerLevel: 70,
                    achievements: ['Time Traveler', 'Forecast Fortune-teller', 'Temporal Titan']
                },
                {
                    id: 'recommendation_systems',
                    name: 'Recommendation Systems',
                    description: 'Collaborative filtering and content-based recommendations',
                    prerequisites: ['ml_fundamentals'],
                    maxLevel: 8,
                    pointsPerLevel: 60,
                    achievements: ['Recommendation Royalty', 'Filter Fairy', 'Preference Prophet']
                }
            ]
        },
        {
            id: 'creativity',
            name: 'Creativity',
            icon: Lightbulb,
            color: 'from-pink-500 to-pink-600',
            lightColor: 'bg-pink-100',
            position: { x: -200, y: 100 },
            skills: [
                {
                    id: 'problem_solving',
                    name: 'Problem Solving',
                    description: 'Breaking down complex problems and finding innovative solutions',
                    prerequisites: [],
                    maxLevel: 8,
                    pointsPerLevel: 35,
                    achievements: ['Problem Pioneer', 'Solution Sage', 'Creative Catalyst']
                },
                {
                    id: 'experimental_design',
                    name: 'Experimental Design',
                    description: 'A/B testing, hypothesis formation, and scientific method',
                    prerequisites: ['problem_solving'],
                    maxLevel: 10,
                    pointsPerLevel: 55,
                    achievements: ['Experiment Expert', 'Hypothesis Hero', 'Design Dynamo']
                },
                {
                    id: 'innovation',
                    name: 'Innovation',
                    description: 'Novel approaches, creative thinking, and breakthrough insights',
                    prerequisites: ['experimental_design'],
                    maxLevel: 12,
                    pointsPerLevel: 75,
                    achievements: ['Innovation Icon', 'Breakthrough Bringer', 'Creative Crown']
                }
            ]
        },
        {
            id: 'persistence',
            name: 'Persistence',
            icon: Zap,
            color: 'from-yellow-500 to-yellow-600',
            lightColor: 'bg-yellow-100',
            position: { x: 0, y: 200 },
            skills: [
                {
                    id: 'consistency',
                    name: 'Consistency',
                    description: 'Daily learning habits and regular practice',
                    prerequisites: [],
                    maxLevel: 15,
                    pointsPerLevel: 20,
                    achievements: ['Consistency King', 'Habit Hero', 'Steady Scholar']
                },
                {
                    id: 'resilience',
                    name: 'Resilience',
                    description: 'Overcoming failures and learning from mistakes',
                    prerequisites: ['consistency'],
                    maxLevel: 10,
                    pointsPerLevel: 45,
                    achievements: ['Resilience Ranger', 'Comeback Champion', 'Grit Guardian']
                },
                {
                    id: 'mastery_mindset',
                    name: 'Mastery Mindset',
                    description: 'Deep understanding and perfectionist approach',
                    prerequisites: ['resilience'],
                    maxLevel: 12,
                    pointsPerLevel: 65,
                    achievements: ['Mastery Master', 'Perfectionist Pro', 'Excellence Emperor']
                }
            ]
        }
    ], []);

    // Calculate skill levels and progress
    const skillProgress = useMemo(() => {
        if (!skillData?.data?.skillDistribution) return {};

        const distribution = skillData.data.skillDistribution;
        const progress = {};

        skillCategories.forEach(category => {
            const categoryPoints = distribution.find(d => d.category === category.id)?.total_points || 0;
            progress[category.id] = {
                totalPoints: categoryPoints,
                level: Math.floor(categoryPoints / 100) + 1, // 100 points per category level
                skills: {}
            };

            category.skills.forEach(skill => {
                // Estimate skill points based on category distribution and prerequisites
                const prerequisitesMet = skill.prerequisites.every(prereq => 
                    progress[category.id]?.skills?.[prereq]?.level >= 3
                );
                
                const basePoints = prerequisitesMet ? Math.floor(categoryPoints / category.skills.length) : 0;
                const skillLevel = Math.min(Math.floor(basePoints / skill.pointsPerLevel) + 1, skill.maxLevel);
                const pointsInCurrentLevel = basePoints % skill.pointsPerLevel;
                
                progress[category.id].skills[skill.id] = {
                    points: basePoints,
                    level: skillLevel,
                    pointsInCurrentLevel,
                    pointsToNextLevel: skill.pointsPerLevel - pointsInCurrentLevel,
                    isUnlocked: prerequisitesMet || skill.prerequisites.length === 0,
                    progress: pointsInCurrentLevel / skill.pointsPerLevel,
                    achievements: skill.achievements.slice(0, Math.floor(skillLevel / (skill.maxLevel / skill.achievements.length)))
                };
            });
        });

        return progress;
    }, [skillData, skillCategories]);

    // Get skill node by ID
    const getSkillById = (skillId) => {
        for (const category of skillCategories) {
            const skill = category.skills.find(s => s.id === skillId);
            if (skill) {
                return {
                    ...skill,
                    category: category.id,
                    categoryName: category.name,
                    categoryIcon: category.icon,
                    categoryColor: category.color
                };
            }
        }
        return null;
    };

    // Check if skill is accessible
    const isSkillAccessible = (categoryId, skillId) => {
        return skillProgress[categoryId]?.skills?.[skillId]?.isUnlocked || false;
    };

    // Get skill level
    const getSkillLevel = (categoryId, skillId) => {
        return skillProgress[categoryId]?.skills?.[skillId]?.level || 1;
    };

    // Render skill node
    const renderSkillNode = (category, skill, position) => {
        const isAccessible = isSkillAccessible(category.id, skill.id);
        const skillLevel = getSkillLevel(category.id, skill.id);
        const skillData = skillProgress[category.id]?.skills?.[skill.id];
        const isHovered = hoveredSkill === `${category.id}-${skill.id}`;
        const isSelected = selectedSkill?.id === skill.id;

        const IconComponent = category.icon;

        // Skip if filtered
        if (skillFilter === 'unlocked' && !isAccessible) return null;
        if (skillFilter === 'locked' && isAccessible) return null;

        return (
            <motion.div
                key={`${category.id}-${skill.id}`}
                layout
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ 
                    opacity: 1, 
                    scale: isHovered ? 1.1 : isSelected ? 1.05 : 1,
                    x: position.x * zoom,
                    y: position.y * zoom
                }}
                transition={{ duration: 0.3 }}
                className={`
                    absolute cursor-pointer select-none
                    ${isAccessible ? 'z-10' : 'z-5'}
                `}
                style={{
                    left: '50%',
                    top: '50%',
                    transform: `translate(-50%, -50%) translate(${position.x * zoom}px, ${position.y * zoom}px)`
                }}
                onClick={() => setSelectedSkill(skill)}
                onMouseEnter={() => setHoveredSkill(`${category.id}-${skill.id}`)}
                onMouseLeave={() => setHoveredSkill(null)}
            >
                {/* Skill Node */}
                <div className={`
                    relative w-16 h-16 rounded-full border-2 transition-all duration-300
                    ${isAccessible 
                        ? `bg-gradient-to-br ${category.color} border-white shadow-lg` 
                        : 'bg-gray-600 border-gray-500'
                    }
                    ${isHovered ? 'shadow-2xl' : ''}
                    ${isSelected ? 'ring-2 ring-yellow-400' : ''}
                `}>
                    {/* Icon */}
                    <div className="absolute inset-0 flex items-center justify-center">
                        {isAccessible ? (
                            <IconComponent className="w-8 h-8 text-white" />
                        ) : (
                            <Lock className="w-6 h-6 text-gray-400" />
                        )}
                    </div>

                    {/* Level indicator */}
                    {isAccessible && (
                        <div className="absolute -bottom-1 -right-1 w-6 h-6 bg-yellow-500 rounded-full border-2 border-white flex items-center justify-center">
                            <span className="text-xs font-bold text-white">{skillLevel}</span>
                        </div>
                    )}

                    {/* Achievement badge */}
                    {isAccessible && skillData?.achievements.length > 0 && (
                        <div className="absolute -top-1 -right-1">
                            <Trophy className="w-4 h-4 text-yellow-400" />
                        </div>
                    )}

                    {/* Progress ring */}
                    {isAccessible && skillData?.progress > 0 && (
                        <svg className="absolute inset-0 w-full h-full" style={{ transform: 'rotate(-90deg)' }}>
                            <circle
                                cx="50%"
                                cy="50%"
                                r="30"
                                stroke="rgba(255,255,255,0.3)"
                                strokeWidth="2"
                                fill="none"
                            />
                            <circle
                                cx="50%"
                                cy="50%"
                                r="30"
                                stroke="white"
                                strokeWidth="2"
                                fill="none"
                                strokeDasharray={`${skillData.progress * 188.4} 188.4`}
                                className="transition-all duration-500"
                            />
                        </svg>
                    )}
                </div>

                {/* Skill Name */}
                <div className="absolute top-20 left-1/2 transform -translate-x-1/2 text-center min-w-max">
                    <p className={`text-xs font-medium ${isAccessible ? 'text-white' : 'text-gray-500'}`}>
                        {skill.name}
                    </p>
                    {isAccessible && (
                        <p className="text-xs text-gray-400">
                            Level {skillLevel}/{skill.maxLevel}
                        </p>
                    )}
                </div>

                {/* Hover tooltip */}
                {isHovered && (
                    <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="absolute top-24 left-1/2 transform -translate-x-1/2 bg-gray-900 border border-gray-700 rounded-lg p-3 min-w-64 z-50"
                    >
                        <h4 className="font-semibold text-white mb-1">{skill.name}</h4>
                        <p className="text-sm text-gray-400 mb-2">{skill.description}</p>
                        
                        {isAccessible ? (
                            <div className="space-y-2">
                                <div className="flex justify-between text-xs">
                                    <span className="text-gray-400">Progress</span>
                                    <span className="text-white">
                                        {skillData?.pointsInCurrentLevel || 0}/{skill.pointsPerLevel}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-700 rounded-full h-1">
                                    <div 
                                        className="bg-white h-1 rounded-full transition-all duration-300"
                                        style={{ width: `${(skillData?.progress || 0) * 100}%` }}
                                    />
                                </div>
                                {skillData?.achievements.length > 0 && (
                                    <div className="pt-2 border-t border-gray-700">
                                        <p className="text-xs text-gray-400 mb-1">Achievements:</p>
                                        <div className="flex flex-wrap gap-1">
                                            {skillData.achievements.map((achievement, index) => (
                                                <span key={index} className="text-xs bg-yellow-500 bg-opacity-20 text-yellow-400 px-2 py-1 rounded">
                                                    {achievement}
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        ) : (
                            <div className="space-y-2">
                                <p className="text-xs text-red-400">Prerequisites required:</p>
                                <div className="space-y-1">
                                    {skill.prerequisites.map(prereqId => {
                                        const prereqSkill = getSkillById(prereqId);
                                        return prereqSkill && (
                                            <div key={prereqId} className="text-xs text-gray-400 flex items-center space-x-1">
                                                <Circle className="w-3 h-3" />
                                                <span>{prereqSkill.name} (Level 3+)</span>
                                            </div>
                                        );
                                    })}
                                </div>
                            </div>
                        )}
                    </motion.div>
                )}
            </motion.div>
        );
    };

    // Render connections between skills
    const renderConnections = () => {
        if (!showConnections) return null;

        const connections = [];
        
        skillCategories.forEach(category => {
            category.skills.forEach(skill => {
                skill.prerequisites.forEach(prereqId => {
                    const prereqSkill = getSkillById(prereqId);
                    if (prereqSkill) {
                        // Find positions for both skills
                        const prereqCategory = skillCategories.find(c => c.skills.some(s => s.id === prereqId));
                        const skillIndex = category.skills.findIndex(s => s.id === skill.id);
                        const prereqIndex = prereqCategory?.skills.findIndex(s => s.id === prereqId);

                        if (prereqCategory && skillIndex !== -1 && prereqIndex !== -1) {
                            const fromPos = {
                                x: prereqCategory.position.x + (prereqIndex - prereqCategory.skills.length / 2) * 80,
                                y: prereqCategory.position.y + 40
                            };
                            const toPos = {
                                x: category.position.x + (skillIndex - category.skills.length / 2) * 80,
                                y: category.position.y - 40
                            };

                            connections.push(
                                <line
                                    key={`${prereqId}-${skill.id}`}
                                    x1={fromPos.x * zoom + 400}
                                    y1={fromPos.y * zoom + 300}
                                    x2={toPos.x * zoom + 400}
                                    y2={toPos.y * zoom + 300}
                                    stroke="rgba(255,255,255,0.2)"
                                    strokeWidth="2"
                                    strokeDasharray={isSkillAccessible(category.id, skill.id) ? "0" : "5,5"}
                                />
                            );
                        }
                    }
                });
            });
        });

        return (
            <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {connections}
            </svg>
        );
    };

    // Render category clusters
    const renderCategoryCluster = (category) => {
        return (
            <div key={category.id}>
                {/* Category label */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="absolute"
                    style={{
                        left: '50%',
                        top: '50%',
                        transform: `translate(-50%, -50%) translate(${category.position.x * zoom}px, ${(category.position.y - 80) * zoom}px)`
                    }}
                >
                    <div className={`
                        px-4 py-2 rounded-lg bg-gradient-to-r ${category.color} text-white text-sm font-semibold
                        flex items-center space-x-2 shadow-lg
                    `}>
                        <category.icon className="w-4 h-4" />
                        <span>{category.name}</span>
                        <div className="bg-white bg-opacity-20 px-2 py-1 rounded text-xs">
                            Lv. {skillProgress[category.id]?.level || 1}
                        </div>
                    </div>
                </motion.div>

                {/* Skills in category */}
                {category.skills.map((skill, index) => {
                    const skillPosition = {
                        x: category.position.x + (index - category.skills.length / 2) * 80,
                        y: category.position.y
                    };
                    return renderSkillNode(category, skill, skillPosition);
                })}
            </div>
        );
    };

    // Render skill details modal
    const renderSkillModal = () => {
        if (!selectedSkill) return null;

        const skillCategory = skillCategories.find(c => c.skills.some(s => s.id === selectedSkill.id));
        const skillData = skillProgress[skillCategory?.id]?.skills?.[selectedSkill.id];
        const isAccessible = isSkillAccessible(skillCategory?.id, selectedSkill.id);

        return (
            <AnimatePresence>
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
                    onClick={() => setSelectedSkill(null)}
                >
                    <motion.div
                        initial={{ scale: 0.9, opacity: 0 }}
                        animate={{ scale: 1, opacity: 1 }}
                        exit={{ scale: 0.9, opacity: 0 }}
                        className="bg-gray-900 rounded-xl p-6 max-w-md w-full border border-gray-700"
                        onClick={e => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="flex items-start justify-between mb-4">
                            <div className="flex items-center space-x-3">
                                <div className={`
                                    p-3 rounded-lg bg-gradient-to-br ${skillCategory?.color}
                                    ${isAccessible ? '' : 'grayscale opacity-50'}
                                `}>
                                    {isAccessible ? (
                                        <skillCategory.icon className="w-6 h-6 text-white" />
                                    ) : (
                                        <Lock className="w-6 h-6 text-white" />
                                    )}
                                </div>
                                <div>
                                    <h3 className="text-xl font-bold text-white">{selectedSkill.name}</h3>
                                    <p className="text-sm text-gray-400">{skillCategory?.name}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setSelectedSkill(null)}
                                className="text-gray-400 hover:text-white"
                            >
                                ×
                            </button>
                        </div>

                        {/* Description */}
                        <p className="text-gray-300 mb-6">{selectedSkill.description}</p>

                        {isAccessible ? (
                            <div className="space-y-4">
                                {/* Progress */}
                                <div>
                                    <div className="flex justify-between items-center mb-2">
                                        <span className="text-sm text-gray-400">Level Progress</span>
                                        <span className="text-sm text-white">
                                            Level {skillData?.level || 1}/{selectedSkill.maxLevel}
                                        </span>
                                    </div>
                                    <div className="w-full bg-gray-700 rounded-full h-2">
                                        <div 
                                            className={`h-2 rounded-full bg-gradient-to-r ${skillCategory?.color} transition-all duration-500`}
                                            style={{ width: `${(skillData?.progress || 0) * 100}%` }}
                                        />
                                    </div>
                                    <div className="flex justify-between text-xs text-gray-400 mt-1">
                                        <span>{skillData?.pointsInCurrentLevel || 0} pts</span>
                                        <span>{selectedSkill.pointsPerLevel} pts</span>
                                    </div>
                                </div>

                                {/* Achievements */}
                                {skillData?.achievements.length > 0 && (
                                    <div>
                                        <h4 className="text-sm font-semibold text-white mb-2">Achievements Unlocked</h4>
                                        <div className="space-y-2">
                                            {skillData.achievements.map((achievement, index) => (
                                                <div key={index} className="flex items-center space-x-2 bg-yellow-500 bg-opacity-10 border border-yellow-500 border-opacity-30 rounded-lg p-2">
                                                    <Trophy className="w-4 h-4 text-yellow-400" />
                                                    <span className="text-sm text-yellow-400">{achievement}</span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {/* Next Steps */}
                                <div className="pt-4 border-t border-gray-700">
                                    <button
                                        onClick={() => {
                                            setSelectedSkill(null);
                                            navigate('/learning');
                                        }}
                                        className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
                                    >
                                        <BookOpen className="w-4 h-4" />
                                        <span>Continue Learning</span>
                                    </button>
                                </div>
                            </div>
                        ) : (
                            <div className="space-y-4">
                                {/* Prerequisites */}
                                <div>
                                    <h4 className="text-sm font-semibold text-white mb-2">Prerequisites Required</h4>
                                    <div className="space-y-2">
                                        {selectedSkill.prerequisites.map(prereqId => {
                                            const prereqSkill = getSkillById(prereqId);
                                            return prereqSkill && (
                                                <div key={prereqId} className="flex items-center space-x-2 bg-red-500 bg-opacity-10 border border-red-500 border-opacity-30 rounded-lg p-2">
                                                    <Lock className="w-4 h-4 text-red-400" />
                                                    <span className="text-sm text-red-400">{prereqSkill.name} (Level 3+)</span>
                                                </div>
                                            );
                                        })}
                                    </div>
                                </div>

                                {/* Action */}
                                <div className="pt-4 border-t border-gray-700">
                                    <button
                                        onClick={() => {
                                            setSelectedSkill(null);
                                            navigate('/learning');
                                        }}
                                        className="w-full bg-gray-600 hover:bg-gray-700 text-white py-2 px-4 rounded-lg transition-colors flex items-center justify-center space-x-2"
                                    >
                                        <BookOpen className="w-4 h-4" />
                                        <span>Work on Prerequisites</span>
                                    </button>
                                </div>
                            </div>
                        )}
                    </motion.div>
                </motion.div>
            </AnimatePresence>
        );
    };

    // Loading state
    if (isLoading) {
        return (
            <div className={`relative w-full h-96 bg-gray-800 rounded-lg flex items-center justify-center ${className}`}>
                <div className="text-center">
                    <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                        className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"
                    />
                    <p className="text-gray-400">Loading skill tree...</p>
                </div>
            </div>
        );
    }

    return (
        <div className={`relative ${className}`}>
            {/* Controls */}
            <div className="flex items-center justify-between mb-6">
                <div>
                    <h2 className="text-2xl font-bold text-white mb-2">Skill Tree</h2>
                    <p className="text-gray-400">Track your mastery across all learning domains</p>
                </div>
                
                <div className="flex items-center space-x-2">
                    {/* Filter */}
                    <select 
                        value={skillFilter}
                        onChange={(e) => setSkillFilter(e.target.value)}
                        className="bg-gray-700 text-white px-3 py-2 rounded-lg border border-gray-600 text-sm"
                    >
                        <option value="all">All Skills</option>
                        <option value="unlocked">Unlocked Only</option>
                        <option value="locked">Locked Only</option>
                    </select>

                    {/* Toggle Connections */}
                    <button
                        onClick={() => setShowConnections(!showConnections)}
                        className={`p-2 rounded-lg transition-colors ${
                            showConnections 
                                ? 'bg-blue-600 text-white' 
                                : 'bg-gray-700 text-gray-400 hover:text-white'
                        }`}
                    >
                        {showConnections ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    </button>

                    {/* Zoom Controls */}
                    <div className="flex items-center space-x-1 bg-gray-700 rounded-lg p-1">
                        <button
                            onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                            className="p-1 hover:bg-gray-600 rounded"
                        >
                            <Minus className="w-4 h-4 text-white" />
                        </button>
                        <span className="text-xs text-gray-400 px-2">{Math.round(zoom * 100)}%</span>
                        <button
                            onClick={() => setZoom(Math.min(2, zoom + 0.1))}
                            className="p-1 hover:bg-gray-600 rounded"
                        >
                            <Plus className="w-4 h-4 text-white" />
                        </button>
                    </div>

                    {/* Reset */}
                    <button
                        onClick={() => {
                            setZoom(1);
                            setSelectedSkill(null);
                            setHoveredSkill(null);
                        }}
                        className="p-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
                    >
                        <RotateCcw className="w-4 h-4" />
                    </button>
                </div>
            </div>

            {/* Skill Tree Container */}
            <div 
                ref={containerRef}
                className="relative w-full h-96 bg-gray-900 rounded-lg border border-gray-700 overflow-hidden"
            >
                {/* Connections */}
                {renderConnections()}

                {/* Category Clusters */}
                {skillCategories.map(category => renderCategoryCluster(category))}
            </div>

            {/* Skill Details Modal */}
            {renderSkillModal()}

            {/* Legend */}
            <div className="mt-6 grid grid-cols-2 lg:grid-cols-6 gap-4">
                {skillCategories.map(category => {
                    const categoryProgress = skillProgress[category.id];
                    const IconComponent = category.icon;
                    
                    return (
                        <div key={category.id} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                            <div className="flex items-center space-x-2 mb-2">
                                <div className={`p-1 rounded bg-gradient-to-br ${category.color}`}>
                                    <IconComponent className="w-3 h-3 text-white" />
                                </div>
                                <span className="text-sm font-medium text-white">{category.name}</span>
                            </div>
                            <div className="text-xs text-gray-400">
                                Level {categoryProgress?.level || 1} • {categoryProgress?.totalPoints || 0} pts
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
};

export default SkillTree;