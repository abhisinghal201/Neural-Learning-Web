/**
 * Enhanced Neural Odyssey Code Playground Component
 *
 * Now fully leverages ALL backend quest submission capabilities:
 * - Rich quest metadata submission with timing data
 * - Code quality analysis and mentor feedback integration
 * - Comprehensive execution result tracking
 * - Self-reflection and learning insights capture
 * - Multiple attempts tracking with improvement metrics
 * - Test case validation with detailed feedback
 * - Code persistence with version history
 * - Integration with skill point award system
 * - Real-time code execution with Pyodide
 * - Advanced code editor with Monaco
 * - Performance metrics and execution analytics
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQueryClient } from 'react-query';
import {
  Play,
  Pause,
  Square,
  RotateCcw,
  Save,
  Download,
  Upload,
  Code,
  Terminal,
  Eye,
  EyeOff,
  Maximize2,
  Minimize2,
  Settings,
  HelpCircle,
  CheckCircle,
  AlertCircle,
  Clock,
  Zap,
  Brain,
  Target,
  Award,
  Star,
  TrendingUp,
  BarChart3,
  MessageSquare,
  Lightbulb,
  RefreshCw,
  X,
  Check,
  ChevronRight,
  ChevronDown,
  ChevronUp,
  FileText,
  Clipboard,
  Timer,
  Activity,
  Layers,
  GitBranch,
  Database,
  Cpu,
  Memory,
  Loader
} from 'lucide-react';
import toast from 'react-hot-toast';

// Utils
import { api } from '../utils/api';

const CodePlayground = ({
  questId,
  questTitle,
  questType = 'coding_exercise',
  phase,
  week,
  difficultyLevel = 1,
  testCases = [],
  initialCode = '',
  language = 'python',
  readOnly = false,
  showTests = true,
  isFullscreen: externalFullscreen = false,
  onToggleFullscreen,
  onCodeChange,
  onQuestComplete,
  className = ''
}) => {
  // State management
  const [code, setCode] = useState(initialCode);
  const [output, setOutput] = useState('');
  const [error, setError] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [pyodideReady, setPyodideReady] = useState(false);
  const [pyodideInstance, setPyodideInstance] = useState(null);
  const [executionTime, setExecutionTime] = useState(0);
  const [memoryUsage, setMemoryUsage] = useState(0);
  const [testResults, setTestResults] = useState([]);
  const [allTestsPassed, setAllTestsPassed] = useState(false);
  
  // Quest submission state
  const [questAttempts, setQuestAttempts] = useState(0);
  const [startTime, setStartTime] = useState(Date.now());
  const [codeHistory, setCodeHistory] = useState([]);
  const [executionHistory, setExecutionHistory] = useState([]);
  const [showSubmissionModal, setShowSubmissionModal] = useState(false);
  const [submissionData, setSubmissionData] = useState({
    self_reflection: '',
    challenges_faced: '',
    key_learnings: '',
    improvement_areas: '',
    confidence_level: 8,
    time_spent_debugging: 0,
    help_resources_used: [],
    code_quality_self_assessment: 8
  });

  // UI state
  const [activeTab, setActiveTab] = useState('output');
  const [isFullscreen, setIsFullscreen] = useState(externalFullscreen);
  const [showSettings, setShowSettings] = useState(false);
  const [fontSize, setFontSize] = useState(14);
  const [theme, setTheme] = useState('vs-dark');
  const [wordWrap, setWordWrap] = useState(true);
  const [autoSave, setAutoSave] = useState(true);
  const [showCodeHistory, setShowCodeHistory] = useState(false);
  const [selectedHistoryIndex, setSelectedHistoryIndex] = useState(-1);

  // Performance tracking
  const [performanceMetrics, setPerformanceMetrics] = useState({
    execution_count: 0,
    total_execution_time: 0,
    average_execution_time: 0,
    memory_peak: 0,
    error_count: 0,
    success_rate: 0
  });

  // Refs
  const editorRef = useRef(null);
  const monacoRef = useRef(null);
  const executionTimeRef = useRef(null);
  const autoSaveTimeoutRef = useRef(null);

  const queryClient = useQueryClient();

  // Quest submission mutation
  const submitQuestMutation = useMutation(
    (questData) => api.learning.submitQuest(questData),
    {
      onSuccess: (data) => {
        queryClient.invalidateQueries(['quests', phase, week]);
        toast.success(`Quest submitted! ${data.data.mentor_feedback || 'Great work!'}`);
        if (onQuestComplete) onQuestComplete(data.data);
        setShowSubmissionModal(false);
      },
      onError: (error) => {
        toast.error('Failed to submit quest: ' + error.message);
      }
    }
  );

  // Auto-save mutation
  const autoSaveMutation = useMutation(
    (saveData) => api.learning.updateQuest(questId, saveData),
    {
      onSuccess: () => {
        // Silent success for auto-save
      },
      onError: () => {
        // Silent error for auto-save
      }
    }
  );

  // Initialize Pyodide
  useEffect(() => {
    const initPyodide = async () => {
      try {
        if (window.pyodide) {
          setPyodideInstance(window.pyodide);
          setPyodideReady(true);
          return;
        }

        // Load Pyodide
        const script = document.createElement('script');
        script.src = 'https://cdnjs.cloudflare.com/ajax/libs/pyodide/0.24.1/pyodide.min.js';
        script.onload = async () => {
          try {
            const pyodide = await window.loadPyodide({
              indexURL: 'https://cdnjs.cloudflare.com/ajax/libs/pyodide/0.24.1/'
            });
            
            // Install common packages
            await pyodide.loadPackage(['numpy', 'matplotlib', 'pandas']);
            
            window.pyodide = pyodide;
            setPyodideInstance(pyodide);
            setPyodideReady(true);
            toast.success('Python environment ready! 🐍');
          } catch (error) {
            console.error('Failed to initialize Pyodide:', error);
            toast.error('Failed to initialize Python environment');
          }
        };
        document.head.appendChild(script);
      } catch (error) {
        console.error('Error loading Pyodide:', error);
        toast.error('Failed to load Python environment');
      }
    };

    initPyodide();
  }, []);

  // Auto-save effect
  useEffect(() => {
    if (autoSave && code !== initialCode && questId) {
      clearTimeout(autoSaveTimeoutRef.current);
      autoSaveTimeoutRef.current = setTimeout(() => {
        const saveData = {
          code_solution: code,
          attempts_count: questAttempts,
          time_spent_minutes: Math.floor((Date.now() - startTime) / (1000 * 60)),
          last_saved: new Date().toISOString()
        };
        autoSaveMutation.mutate(saveData);
      }, 2000);
    }

    return () => clearTimeout(autoSaveTimeoutRef.current);
  }, [code, autoSave, questId, questAttempts, startTime, initialCode]);

  // Code change tracking
  useEffect(() => {
    if (onCodeChange) {
      onCodeChange(code);
    }

    // Track code history
    if (code && code !== initialCode) {
      setCodeHistory(prev => {
        const newHistory = [...prev];
        const timestamp = Date.now();
        newHistory.push({
          code,
          timestamp,
          length: code.length,
          lines: code.split('\n').length
        });
        // Keep only last 50 versions
        return newHistory.slice(-50);
      });
    }
  }, [code, onCodeChange, initialCode]);

  // Run code function
  const runCode = useCallback(async () => {
    if (!pyodideReady || !pyodideInstance || isRunning) return;

    setIsRunning(true);
    setError('');
    setOutput('');
    
    const executionStart = performance.now();
    const memoryBefore = performance.memory ? performance.memory.usedJSHeapSize : 0;

    try {
      // Clear previous output
      pyodideInstance.runPython(`
import sys
import io
import contextlib

# Capture stdout
captured_output = io.StringIO()
sys.stdout = captured_output
      `);

      // Run the user code
      const result = pyodideInstance.runPython(code);
      
      // Get captured output
      const capturedOutput = pyodideInstance.runPython(`
sys.stdout = sys.__stdout__
captured_output.getvalue()
      `);

      const executionEnd = performance.now();
      const executionTimeMs = executionEnd - executionStart;
      const memoryAfter = performance.memory ? performance.memory.usedJSHeapSize : 0;
      const memoryUsed = memoryAfter - memoryBefore;

      setExecutionTime(executionTimeMs);
      setMemoryUsage(memoryUsed);
      setOutput(capturedOutput || (result !== undefined ? String(result) : ''));

      // Update performance metrics
      setPerformanceMetrics(prev => {
        const newCount = prev.execution_count + 1;
        const newTotalTime = prev.total_execution_time + executionTimeMs;
        return {
          execution_count: newCount,
          total_execution_time: newTotalTime,
          average_execution_time: newTotalTime / newCount,
          memory_peak: Math.max(prev.memory_peak, memoryUsed),
          error_count: prev.error_count,
          success_rate: ((newCount - prev.error_count) / newCount) * 100
        };
      });

      // Track execution history
      setExecutionHistory(prev => [...prev, {
        timestamp: Date.now(),
        code: code,
        output: capturedOutput,
        execution_time: executionTimeMs,
        memory_usage: memoryUsed,
        success: true
      }].slice(-20));

      // Run test cases if available
      if (testCases && testCases.length > 0) {
        await runTestCases();
      }

      setQuestAttempts(prev => prev + 1);

    } catch (error) {
      const executionEnd = performance.now();
      const executionTimeMs = executionEnd - executionStart;
      
      setError(error.message);
      setExecutionTime(executionTimeMs);

      // Update error metrics
      setPerformanceMetrics(prev => {
        const newCount = prev.execution_count + 1;
        const newErrorCount = prev.error_count + 1;
        const newTotalTime = prev.total_execution_time + executionTimeMs;
        return {
          ...prev,
          execution_count: newCount,
          total_execution_time: newTotalTime,
          average_execution_time: newTotalTime / newCount,
          error_count: newErrorCount,
          success_rate: ((newCount - newErrorCount) / newCount) * 100
        };
      });

      // Track failed execution
      setExecutionHistory(prev => [...prev, {
        timestamp: Date.now(),
        code: code,
        error: error.message,
        execution_time: executionTimeMs,
        success: false
      }].slice(-20));

      setQuestAttempts(prev => prev + 1);
    } finally {
      setIsRunning(false);
    }
  }, [pyodideReady, pyodideInstance, isRunning, code, testCases]);

  // Run test cases
  const runTestCases = async () => {
    if (!testCases || testCases.length === 0) return;

    const results = [];
    let allPassed = true;

    for (let i = 0; i < testCases.length; i++) {
      const testCase = testCases[i];
      try {
        // Prepare test environment
        pyodideInstance.runPython(`
import sys
import io
test_output = io.StringIO()
sys.stdout = test_output
        `);

        // Run test case
        if (testCase.setup) {
          pyodideInstance.runPython(testCase.setup);
        }

        const testResult = pyodideInstance.runPython(testCase.test_code || testCase.code);
        
        const testOutput = pyodideInstance.runPython(`
sys.stdout = sys.__stdout__
test_output.getvalue()
        `);

        const passed = testCase.expected_output 
          ? testOutput.trim() === testCase.expected_output.trim()
          : true; // If no expected output, assume passed if no error

        results.push({
          ...testCase,
          passed,
          actual_output: testOutput,
          execution_time: performance.now()
        });

        if (!passed) allPassed = false;

      } catch (error) {
        results.push({
          ...testCase,
          passed: false,
          error: error.message,
          execution_time: performance.now()
        });
        allPassed = false;
      }
    }

    setTestResults(results);
    setAllTestsPassed(allPassed);

    if (allPassed) {
      toast.success('All tests passed! 🎉');
    } else {
      toast.error('Some tests failed. Check the test results.');
    }
  };

  // Submit quest
  const submitQuest = () => {
    if (!allTestsPassed && testCases.length > 0) {
      toast.error('Please ensure all tests pass before submitting');
      return;
    }

    setShowSubmissionModal(true);
  };

  // Handle quest submission
  const handleQuestSubmission = () => {
    const timeSpent = Math.floor((Date.now() - startTime) / (1000 * 60));
    
    const questData = {
      quest_id: questId,
      quest_title: questTitle,
      quest_type: questType,
      phase: parseInt(phase),
      week: parseInt(week),
      difficulty_level: difficultyLevel,
      code_solution: code,
      execution_result: {
        output,
        error,
        execution_time: executionTime,
        memory_usage: memoryUsage,
        test_results: testResults,
        all_tests_passed: allTestsPassed,
        performance_metrics: performanceMetrics
      },
      time_to_complete_minutes: timeSpent,
      attempts_count: questAttempts,
      self_reflection: submissionData.self_reflection,
      status: allTestsPassed ? 'completed' : 'attempted',
      completion_metadata: {
        challenges_faced: submissionData.challenges_faced,
        key_learnings: submissionData.key_learnings,
        improvement_areas: submissionData.improvement_areas,
        confidence_level: submissionData.confidence_level,
        time_spent_debugging: submissionData.time_spent_debugging,
        help_resources_used: submissionData.help_resources_used,
        code_quality_self_assessment: submissionData.code_quality_self_assessment,
        code_history_length: codeHistory.length,
        execution_history_length: executionHistory.length,
        final_code_metrics: {
          lines_of_code: code.split('\n').length,
          character_count: code.length,
          complexity_estimate: calculateCodeComplexity(code)
        }
      }
    };

    submitQuestMutation.mutate(questData);
  };

  // Calculate code complexity (simple heuristic)
  const calculateCodeComplexity = (code) => {
    const lines = code.split('\n').filter(line => line.trim());
    const controlStructures = (code.match(/(if|for|while|try|except|with|def|class)/g) || []).length;
    const operators = (code.match(/[+\-*/%=<>!&|]/g) || []).length;
    
    return Math.min(10, Math.max(1, Math.floor((lines.length + controlStructures * 2 + operators * 0.5) / 10)));
  };

  // Toggle fullscreen
  const toggleFullscreen = () => {
    if (onToggleFullscreen) {
      onToggleFullscreen();
    } else {
      setIsFullscreen(!isFullscreen);
    }
  };

  // Load code from history
  const loadFromHistory = (historyItem) => {
    setCode(historyItem.code);
    setSelectedHistoryIndex(codeHistory.findIndex(h => h.timestamp === historyItem.timestamp));
    setShowCodeHistory(false);
    toast.success('Code loaded from history');
  };

  // Reset code
  const resetCode = () => {
    setCode(initialCode);
    setOutput('');
    setError('');
    setTestResults([]);
    setAllTestsPassed(false);
    toast.success('Code reset to initial state');
  };

  // Download code
  const downloadCode = () => {
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${questId || 'code'}.py`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Monaco Editor setup
  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;
    monacoRef.current = monaco;

    // Configure editor
    editor.updateOptions({
      fontSize: fontSize,
      wordWrap: wordWrap ? 'on' : 'off',
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      automaticLayout: true
    });

    // Set initial value
    editor.setValue(code);

    // Listen for changes
    editor.onDidChangeModelContent(() => {
      setCode(editor.getValue());
    });
  };

  // Render Monaco Editor
  const renderEditor = () => {
    const MonacoEditor = React.lazy(() => import('@monaco-editor/react'));

    return (
      <React.Suspense fallback={<div className="flex items-center justify-center h-64"><Loader className="w-8 h-8 animate-spin text-blue-400" /></div>}>
        <MonacoEditor
          height="100%"
          language={language}
          theme={theme}
          value={code}
          onChange={(value) => setCode(value || '')}
          onMount={handleEditorDidMount}
          options={{
            fontSize: fontSize,
            wordWrap: wordWrap ? 'on' : 'off',
            minimap: { enabled: false },
            scrollBeyondLastLine: false,
            automaticLayout: true,
            readOnly: readOnly
          }}
        />
      </React.Suspense>
    );
  };

  // Render output panel
  const renderOutput = () => {
    return (
      <div className="h-full flex flex-col">
        <div className="flex-1 overflow-y-auto">
          {output && (
            <div className="p-4">
              <div className="text-xs text-gray-400 mb-2">Output:</div>
              <pre className="text-sm text-green-400 bg-gray-900 p-3 rounded overflow-x-auto">
                {output}
              </pre>
            </div>
          )}
          
          {error && (
            <div className="p-4">
              <div className="text-xs text-gray-400 mb-2">Error:</div>
              <pre className="text-sm text-red-400 bg-gray-900 p-3 rounded overflow-x-auto">
                {error}
              </pre>
            </div>
          )}

          {!output && !error && (
            <div className="p-4 text-center text-gray-500">
              Run your code to see output here
            </div>
          )}
        </div>

        {/* Execution metrics */}
        {(executionTime > 0 || memoryUsage > 0) && (
          <div className="border-t border-gray-700 p-3">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <div className="flex items-center space-x-4">
                {executionTime > 0 && (
                  <div className="flex items-center space-x-1">
                    <Timer className="w-3 h-3" />
                    <span>{executionTime.toFixed(2)}ms</span>
                  </div>
                )}
                {memoryUsage > 0 && (
                  <div className="flex items-center space-x-1">
                    <Memory className="w-3 h-3" />
                    <span>{(memoryUsage / 1024 / 1024).toFixed(2)}MB</span>
                  </div>
                )}
                <div className="flex items-center space-x-1">
                  <Activity className="w-3 h-3" />
                  <span>Attempts: {questAttempts}</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  // Render test results
  const renderTests = () => {
    return (
      <div className="h-full overflow-y-auto">
        {testResults.length > 0 ? (
          <div className="p-4 space-y-3">
            {testResults.map((result, index) => (
              <div
                key={index}
                className={`p-3 rounded-lg border ${
                  result.passed
                    ? 'bg-green-600/10 border-green-600/30'
                    : 'bg-red-600/10 border-red-600/30'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    {result.passed ? (
                      <CheckCircle className="w-4 h-4 text-green-400" />
                    ) : (
                      <AlertCircle className="w-4 h-4 text-red-400" />
                    )}
                    <span className="text-sm font-medium text-white">
                      {result.description || `Test ${index + 1}`}
                    </span>
                  </div>
                  <span className={`text-xs px-2 py-1 rounded ${
                    result.passed ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                  }`}>
                    {result.passed ? 'PASS' : 'FAIL'}
                  </span>
                </div>

                {result.input && (
                  <div className="text-xs text-gray-400 mb-1">
                    Input: <span className="text-gray-300">{result.input}</span>
                  </div>
                )}

                {result.expected_output && (
                  <div className="text-xs text-gray-400 mb-1">
                    Expected: <span className="text-gray-300">{result.expected_output}</span>
                  </div>
                )}

                {result.actual_output && (
                  <div className="text-xs text-gray-400 mb-1">
                    Actual: <span className={result.passed ? 'text-green-300' : 'text-red-300'}>
                      {result.actual_output}
                    </span>
                  </div>
                )}

                {result.error && (
                  <div className="text-xs text-red-400">
                    Error: {result.error}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="p-4 space-y-2">
            {testCases.map((testCase, index) => (
              <div key={index} className="p-3 bg-gray-800 rounded-lg border border-gray-700">
                <div className="text-sm text-white mb-1">
                  {testCase.description || `Test ${index + 1}`}
                </div>
                {testCase.input && (
                  <div className="text-xs text-gray-400">
                    Input: {testCase.input}
                  </div>
                )}
                {testCase.expected_output && (
                  <div className="text-xs text-gray-400">
                    Expected: {testCase.expected_output}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  // Render performance metrics
  const renderMetrics = () => {
    return (
      <div className="h-full overflow-y-auto p-4 space-y-4">
        <div>
          <h4 className="text-sm font-medium text-white mb-3">Performance Metrics</h4>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="text-xs text-gray-400">Executions</div>
              <div className="text-lg font-bold text-white">{performanceMetrics.execution_count}</div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="text-xs text-gray-400">Success Rate</div>
              <div className="text-lg font-bold text-green-400">
                {performanceMetrics.success_rate.toFixed(1)}%
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="text-xs text-gray-400">Avg Time</div>
              <div className="text-lg font-bold text-blue-400">
                {performanceMetrics.average_execution_time.toFixed(2)}ms
              </div>
            </div>
            <div className="bg-gray-800 rounded-lg p-3">
              <div className="text-xs text-gray-400">Peak Memory</div>
              <div className="text-lg font-bold text-purple-400">
                {(performanceMetrics.memory_peak / 1024 / 1024).toFixed(2)}MB
              </div>
            </div>
          </div>
        </div>

        {executionHistory.length > 0 && (
          <div>
            <h4 className="text-sm font-medium text-white mb-3">Execution History</h4>
            <div className="space-y-2">
              {executionHistory.slice(-5).reverse().map((execution, index) => (
                <div key={execution.timestamp} className="bg-gray-800 rounded-lg p-3">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-400">
                      {new Date(execution.timestamp).toLocaleTimeString()}
                    </span>
                    <span className={`text-xs px-2 py-1 rounded ${
                      execution.success ? 'bg-green-600 text-white' : 'bg-red-600 text-white'
                    }`}>
                      {execution.success ? 'SUCCESS' : 'ERROR'}
                    </span>
                  </div>
                  <div className="text-xs text-gray-400">
                    Time: {execution.execution_time?.toFixed(2)}ms
                    {execution.memory_usage && (
                      <span className="ml-2">
                        Memory: {(execution.memory_usage / 1024 / 1024).toFixed(2)}MB
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`
      flex flex-col h-full bg-gray-900 rounded-lg border border-gray-700 overflow-hidden
      ${isFullscreen ? 'fixed inset-0 z-50' : ''} ${className}
    `}>
      {/* Toolbar */}
      <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
        <div className="flex items-center space-x-2">
          <Code className="w-4 h-4 text-gray-400" />
          <span className="text-sm font-medium text-white">Code Playground</span>
          <span className="text-xs text-gray-400 capitalize">({language})</span>
          {questTitle && (
            <span className="text-xs text-blue-400">• {questTitle}</span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Run Button */}
          <button
            onClick={runCode}
            disabled={isRunning || !pyodideReady}
            className={`
              flex items-center space-x-1 px-3 py-1 rounded text-sm font-medium transition-colors
              ${isRunning || !pyodideReady
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed' 
                : 'bg-green-600 hover:bg-green-700 text-white'
              }
            `}
          >
            {isRunning ? (
              <Loader className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4" />
            )}
            <span>{isRunning ? 'Running...' : 'Run'}</span>
          </button>

          {/* Reset Button */}
          <button
            onClick={resetCode}
            className="p-1 text-gray-400 hover:text-white"
            title="Reset Code"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          {/* Save Button */}
          <button
            onClick={downloadCode}
            className="p-1 text-gray-400 hover:text-white"
            title="Download Code"
          >
            <Download className="w-4 h-4" />
          </button>

          {/* History Button */}
          <button
            onClick={() => setShowCodeHistory(!showCodeHistory)}
            className="p-1 text-gray-400 hover:text-white"
            title="Code History"
          >
            <GitBranch className="w-4 h-4" />
          </button>

          {/* Settings Button */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-1 text-gray-400 hover:text-white"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>

          {/* Fullscreen Button */}
          <button
            onClick={toggleFullscreen}
            className="p-1 text-gray-400 hover:text-white"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen'}
          >
            {isFullscreen ? (
              <Minimize2 className="w-4 h-4" />
            ) : (
              <Maximize2 className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Code Editor */}
        <div className="flex-1 min-w-0 relative">
          {renderEditor()}
          
          {/* Code History Overlay */}
          {showCodeHistory && (
            <div className="absolute top-0 right-0 w-80 h-full bg-gray-800 border-l border-gray-700 overflow-y-auto">
              <div className="p-4">
                <div className="flex items-center justify-between mb-3">
                  <h4 className="text-sm font-medium text-white">Code History</h4>
                  <button
                    onClick={() => setShowCodeHistory(false)}
                    className="text-gray-400 hover:text-white"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
                <div className="space-y-2">
                  {codeHistory.slice(-10).reverse().map((historyItem, index) => (
                    <div
                      key={historyItem.timestamp}
                      className="p-2 bg-gray-700 rounded cursor-pointer hover:bg-gray-600"
                      onClick={() => loadFromHistory(historyItem)}
                    >
                      <div className="text-xs text-gray-400">
                        {new Date(historyItem.timestamp).toLocaleTimeString()}
                      </div>
                      <div className="text-xs text-white">
                        {historyItem.lines} lines • {historyItem.length} chars
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Side Panel */}
        <div className="w-96 flex flex-col border-l border-gray-700">
          {/* Tab Navigation */}
          <div className="flex bg-gray-800 border-b border-gray-700">
            <button
              onClick={() => setActiveTab('output')}
              className={`
                flex-1 px-4 py-2 text-sm font-medium transition-colors
                ${activeTab === 'output' 
                  ? 'bg-gray-900 text-white border-b-2 border-blue-500' 
                  : 'text-gray-400 hover:text-white'
                }
              `}
            >
              Output
            </button>

            {showTests && testCases && testCases.length > 0 && (
              <button
                onClick={() => setActiveTab('tests')}
                className={`
                  flex-1 px-4 py-2 text-sm font-medium transition-colors
                  ${activeTab === 'tests' 
                    ? 'bg-gray-900 text-white border-b-2 border-blue-500' 
                    : 'text-gray-400 hover:text-white'
                  }
                `}
              >
                Tests
                {testResults.length > 0 && (
                  <span className={`ml-1 px-1 text-xs rounded ${
                    allTestsPassed ? 'bg-green-600' : 'bg-red-600'
                  }`}>
                    {testResults.filter(r => r.passed).length}/{testResults.length}
                  </span>
                )}
              </button>
            )}

            <button
              onClick={() => setActiveTab('metrics')}
              className={`
                flex-1 px-4 py-2 text-sm font-medium transition-colors
                ${activeTab === 'metrics' 
                  ? 'bg-gray-900 text-white border-b-2 border-blue-500' 
                  : 'text-gray-400 hover:text-white'
                }
              `}
            >
              Metrics
            </button>
          </div>

          {/* Tab Content */}
          <div className="flex-1 overflow-hidden">
            {activeTab === 'output' && renderOutput()}
            {activeTab === 'tests' && renderTests()}
            {activeTab === 'metrics' && renderMetrics()}
          </div>
        </div>
      </div>

      {/* Submit Button */}
      {questId && !readOnly && (
        <div className="border-t border-gray-700 p-4">
          <button
            onClick={submitQuest}
            disabled={submitQuestMutation.isLoading || (testCases.length > 0 && !allTestsPassed)}
            className={`
              w-full py-2 px-4 rounded font-medium transition-colors
              ${allTestsPassed || testCases.length === 0
                ? 'bg-blue-600 hover:bg-blue-700 text-white'
                : 'bg-gray-600 cursor-not-allowed text-gray-400'
              }
            `}
          >
            {submitQuestMutation.isLoading ? 'Submitting...' : 'Submit Quest'}
          </button>
        </div>
      )}

      {/* Status Bar */}
      <div className="flex items-center justify-between px-4 py-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-400">
        <div className="flex items-center space-x-4">
          <span>Python {pyodideReady ? 'Ready' : 'Loading...'}</span>
          {executionTime > 0 && (
            <span>Last run: {executionTime.toFixed(2)}ms</span>
          )}
          {questAttempts > 0 && (
            <span>Attempts: {questAttempts}</span>
          )}
        </div>
        <div className="flex items-center space-x-2">
          {autoSaveMutation.isLoading && (
            <span className="flex items-center space-x-1">
              <Loader className="w-3 h-3 animate-spin" />
              <span>Saving...</span>
            </span>
          )}
          {allTestsPassed && testCases.length > 0 && (
            <span className="text-green-400">All tests passed!</span>
          )}
        </div>
      </div>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, x: 300 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 300 }}
            className="absolute top-0 right-0 w-80 h-full bg-gray-800 border-l border-gray-700 overflow-y-auto z-10"
          >
            <div className="p-4">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-sm font-medium text-white">Settings</h4>
                <button
                  onClick={() => setShowSettings(false)}
                  className="text-gray-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-xs text-gray-400 mb-1">Font Size</label>
                  <input
                    type="range"
                    min="10"
                    max="24"
                    value={fontSize}
                    onChange={(e) => setFontSize(Number(e.target.value))}
                    className="w-full"
                  />
                  <div className="text-xs text-gray-500">{fontSize}px</div>
                </div>

                <div>
                  <label className="block text-xs text-gray-400 mb-1">Theme</label>
                  <select
                    value={theme}
                    onChange={(e) => setTheme(e.target.value)}
                    className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-white text-xs"
                  >
                    <option value="vs-dark">Dark</option>
                    <option value="vs">Light</option>
                    <option value="hc-black">High Contrast</option>
                  </select>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Word Wrap</span>
                  <button
                    onClick={() => setWordWrap(!wordWrap)}
                    className={`px-2 py-1 rounded text-xs ${
                      wordWrap ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
                    }`}
                  >
                    {wordWrap ? 'On' : 'Off'}
                  </button>
                </div>

                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Auto Save</span>
                  <button
                    onClick={() => setAutoSave(!autoSave)}
                    className={`px-2 py-1 rounded text-xs ${
                      autoSave ? 'bg-blue-600 text-white' : 'bg-gray-600 text-gray-300'
                    }`}
                  >
                    {autoSave ? 'On' : 'Off'}
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quest Submission Modal */}
      <AnimatePresence>
        {showSubmissionModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="bg-gray-800 rounded-lg border border-gray-700 p-6 max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto"
            >
              <h3 className="text-lg font-semibold text-white mb-4">Submit Quest</h3>
              
              {/* Quest Summary */}
              <div className="mb-4 p-3 bg-gray-700 rounded">
                <div className="text-sm text-gray-300">
                  Quest: <span className="text-white font-medium">{questTitle}</span>
                </div>
                <div className="text-sm text-gray-300">
                  Time spent: <span className="text-white font-medium">
                    {Math.floor((Date.now() - startTime) / (1000 * 60))} minutes
                  </span>
                </div>
                <div className="text-sm text-gray-300">
                  Attempts: <span className="text-white font-medium">{questAttempts}</span>
                </div>
                {testCases.length > 0 && (
                  <div className="text-sm text-gray-300">
                    Tests: <span className={`font-medium ${allTestsPassed ? 'text-green-400' : 'text-red-400'}`}>
                      {testResults.filter(r => r.passed).length}/{testResults.length} passed
                    </span>
                  </div>
                )}
              </div>

              {/* Self Reflection */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Self Reflection *
                </label>
                <textarea
                  value={submissionData.self_reflection}
                  onChange={(e) => setSubmissionData(prev => ({ ...prev, self_reflection: e.target.value }))}
                  placeholder="Reflect on your learning process, what you discovered, and how you solved the problem..."
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={3}
                  required
                />
              </div>

              {/* Challenges Faced */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Challenges Faced
                </label>
                <textarea
                  value={submissionData.challenges_faced}
                  onChange={(e) => setSubmissionData(prev => ({ ...prev, challenges_faced: e.target.value }))}
                  placeholder="What difficulties did you encounter? How did you overcome them?"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={2}
                />
              </div>

              {/* Key Learnings */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Key Learnings
                </label>
                <textarea
                  value={submissionData.key_learnings}
                  onChange={(e) => setSubmissionData(prev => ({ ...prev, key_learnings: e.target.value }))}
                  placeholder="What did you learn from this quest? Any new concepts or techniques?"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm resize-none"
                  rows={2}
                />
              </div>

              {/* Confidence Level */}
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Confidence Level (1-10)
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={submissionData.confidence_level}
                    onChange={(e) => setSubmissionData(prev => ({ ...prev, confidence_level: Number(e.target.value) }))}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{submissionData.confidence_level}</span>
                </div>
              </div>

              {/* Code Quality Self Assessment */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-300 mb-2">
                  Code Quality Self Assessment (1-10)
                </label>
                <div className="flex items-center space-x-2">
                  <input
                    type="range"
                    min="1"
                    max="10"
                    value={submissionData.code_quality_self_assessment}
                    onChange={(e) => setSubmissionData(prev => ({ ...prev, code_quality_self_assessment: Number(e.target.value) }))}
                    className="flex-1"
                  />
                  <span className="text-sm text-white w-8">{submissionData.code_quality_self_assessment}</span>
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => setShowSubmissionModal(false)}
                  className="flex-1 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white"
                >
                  Cancel
                </button>
                <button
                  onClick={handleQuestSubmission}
                  disabled={!submissionData.self_reflection.trim() || submitQuestMutation.isLoading}
                  className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white"
                >
                  {submitQuestMutation.isLoading ? 'Submitting...' : 'Submit Quest'}
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default CodePlayground;