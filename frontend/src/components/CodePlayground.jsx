/**
 * Neural Odyssey Code Playground Component
 *
 * Interactive code editor and execution environment for coding exercises and quests.
 * Supports Python execution via Pyodide, syntax highlighting, test case validation,
 * and integration with the learning progress system.
 *
 * Features:
 * - Monaco Editor with syntax highlighting and IntelliSense
 * - Browser-based Python execution using Pyodide
 * - Automated test case validation and feedback
 * - Real-time code execution with output capture
 * - Error handling and debugging assistance
 * - Code templates and starter code
 * - Hint system and contextual help
 * - Progress saving and code persistence
 * - Performance monitoring and execution limits
 * - Multi-language support (Python focus)
 *
 * Author: Neural Explorer
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import {
    Play,
    Square,
    RotateCcw,
    Save,
    Download,
    Upload,
    Copy,
    Check,
    X,
    AlertCircle,
    CheckCircle,
    Clock,
    Zap,
    Brain,
    Lightbulb,
    Target,
    Eye,
    EyeOff,
    Settings,
    Maximize2,
    Minimize2,
    RefreshCw,
    Terminal,
    Code,
    FileText,
    HelpCircle,
    ChevronRight,
    ChevronDown,
    Loader
} from 'lucide-react';
import { api } from '../utils/api';
import toast from 'react-hot-toast';

// Monaco Editor dynamic import
let monaco;
const loadMonaco = async () => {
    if (!monaco) {
        const monacoEditor = await import('@monaco-editor/react');
        monaco = monacoEditor.default;
    }
    return monaco;
};

const CodePlayground = ({
    questId,
    initialCode = '',
    language = 'python',
    testCases = [],
    className = '',
    onCodeChange,
    onTestComplete,
    readOnly = false,
    showTests = true,
    autoSave = true
}) => {
    const queryClient = useQueryClient();
    const editorRef = useRef(null);
    const pyodideRef = useRef(null);
    const executionTimeoutRef = useRef(null);

    // State management
    const [code, setCode] = useState(initialCode);
    const [output, setOutput] = useState('');
    const [errors, setErrors] = useState([]);
    const [isRunning, setIsRunning] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [testResults, setTestResults] = useState([]);
    const [executionTime, setExecutionTime] = useState(0);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [showHints, setShowHints] = useState(false);
    const [activeTab, setActiveTab] = useState('code'); // 'code' | 'output' | 'tests'
    const [pyodideReady, setPyodideReady] = useState(false);
    const [editorSettings, setEditorSettings] = useState({
        fontSize: 14,
        theme: 'vs-dark',
        wordWrap: 'on',
        minimap: { enabled: false }
    });

    // Auto-save mutation
    const saveCodeMutation = useMutation(
        (codeData) => api.post(`/learning/quests/${questId}/save-code`, codeData),
        {
            onSuccess: () => {
                toast.success('Code saved!');
            },
            onError: (error) => {
                console.error('Failed to save code:', error);
            }
        }
    );

    // Submit solution mutation
    const submitSolutionMutation = useMutation(
        (submissionData) => api.post(`/learning/quests/${questId}/submit`, submissionData),
        {
            onSuccess: (data) => {
                toast.success('Solution submitted successfully!');
                if (onTestComplete) {
                    onTestComplete(data.data);
                }
                queryClient.invalidateQueries(['quest', questId]);
            },
            onError: (error) => {
                toast.error('Failed to submit solution');
                console.error('Submission error:', error);
            }
        }
    );

    // Load Pyodide
    useEffect(() => {
        const initPyodide = async () => {
            try {
                setIsLoading(true);
                
                // Load Pyodide from CDN
                if (!window.pyodide) {
                    const script = document.createElement('script');
                    script.src = 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.js';
                    script.onload = async () => {
                        window.pyodide = await window.loadPyodide({
                            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.24.1/full/'
                        });
                        
                        // Install common packages
                        await window.pyodide.loadPackage(['numpy', 'pandas', 'matplotlib']);
                        
                        pyodideRef.current = window.pyodide;
                        setPyodideReady(true);
                        setIsLoading(false);
                    };
                    document.head.appendChild(script);
                } else {
                    pyodideRef.current = window.pyodide;
                    setPyodideReady(true);
                    setIsLoading(false);
                }
            } catch (error) {
                console.error('Failed to load Pyodide:', error);
                setIsLoading(false);
                toast.error('Failed to initialize Python environment');
            }
        };

        if (language === 'python') {
            initPyodide();
        }
    }, [language]);

    // Auto-save code
    useEffect(() => {
        if (autoSave && questId && code !== initialCode) {
            const timeoutId = setTimeout(() => {
                saveCodeMutation.mutate({ code, language });
            }, 2000);

            return () => clearTimeout(timeoutId);
        }
    }, [code, autoSave, questId, initialCode]);

    // Handle code change
    const handleCodeChange = useCallback((value) => {
        setCode(value || '');
        setErrors([]);
        if (onCodeChange) {
            onCodeChange(value || '');
        }
    }, [onCodeChange]);

    // Execute Python code
    const executePythonCode = async (codeToRun) => {
        if (!pyodideRef.current) {
            throw new Error('Python environment not ready');
        }

        const pyodide = pyodideRef.current;
        
        // Clear previous state
        pyodide.runPython(`
import sys
import io
sys.stdout = io.StringIO()
sys.stderr = io.StringIO()
        `);

        try {
            // Execute the code
            const result = pyodide.runPython(codeToRun);
            
            // Capture output
            const stdout = pyodide.runPython("sys.stdout.getvalue()");
            const stderr = pyodide.runPython("sys.stderr.getvalue()");
            
            return {
                result,
                stdout: stdout || '',
                stderr: stderr || '',
                success: true
            };
        } catch (error) {
            const stderr = pyodide.runPython("sys.stderr.getvalue()");
            return {
                result: null,
                stdout: '',
                stderr: stderr || error.message,
                success: false,
                error: error.message
            };
        }
    };

    // Run code
    const runCode = async () => {
        if (isRunning || !code.trim()) return;
        
        setIsRunning(true);
        setOutput('');
        setErrors([]);
        
        const startTime = Date.now();
        
        try {
            // Set execution timeout
            executionTimeoutRef.current = setTimeout(() => {
                setIsRunning(false);
                setErrors(['Execution timeout: Code took too long to run (>10 seconds)']);
                toast.error('Code execution timeout');
            }, 10000);

            let result;
            
            if (language === 'python') {
                if (!pyodideReady) {
                    throw new Error('Python environment is still loading...');
                }
                result = await executePythonCode(code);
            } else {
                throw new Error(`Language ${language} not supported yet`);
            }

            clearTimeout(executionTimeoutRef.current);
            
            const endTime = Date.now();
            setExecutionTime(endTime - startTime);
            
            if (result.success) {
                setOutput(result.stdout || String(result.result || ''));
                if (result.stderr) {
                    setErrors([result.stderr]);
                }
            } else {
                setErrors([result.stderr || result.error || 'Unknown error occurred']);
            }
            
        } catch (error) {
            clearTimeout(executionTimeoutRef.current);
            setErrors([error.message]);
            console.error('Code execution error:', error);
        } finally {
            setIsRunning(false);
        }
    };

    // Run tests
    const runTests = async () => {
        if (!testCases || testCases.length === 0) {
            toast.error('No test cases available');
            return;
        }

        setIsRunning(true);
        const results = [];

        try {
            for (let i = 0; i < testCases.length; i++) {
                const testCase = testCases[i];
                const testCode = `
${code}

# Test case ${i + 1}
${testCase.setup || ''}
result = ${testCase.call || 'main()'}
print(f"Result: {result}")
`;

                const result = await executePythonCode(testCode);
                
                const passed = result.success && (
                    testCase.expectedOutput ? 
                        result.stdout.includes(testCase.expectedOutput) :
                        result.success
                );

                results.push({
                    index: i,
                    description: testCase.description || `Test ${i + 1}`,
                    passed,
                    output: result.stdout,
                    error: result.stderr,
                    expected: testCase.expectedOutput,
                    input: testCase.input
                });
            }

            setTestResults(results);
            
            const passedCount = results.filter(r => r.passed).length;
            if (passedCount === results.length) {
                toast.success('All tests passed! 🎉');
            } else {
                toast.error(`${passedCount}/${results.length} tests passed`);
            }

        } catch (error) {
            console.error('Test execution error:', error);
            toast.error('Failed to run tests');
        } finally {
            setIsRunning(false);
        }
    };

    // Submit solution
    const submitSolution = async () => {
        if (!code.trim()) {
            toast.error('Please write some code before submitting');
            return;
        }

        // Run tests first if available
        if (testCases && testCases.length > 0) {
            await runTests();
            
            // Check if all tests passed
            const allTestsPassed = testResults.every(result => result.passed);
            if (!allTestsPassed) {
                const confirm = window.confirm(
                    'Not all tests are passing. Do you want to submit anyway?'
                );
                if (!confirm) return;
            }
        }

        submitSolutionMutation.mutate({
            code,
            language,
            testResults,
            executionTime,
            output
        });
    };

    // Reset code
    const resetCode = () => {
        if (window.confirm('Are you sure you want to reset your code? This cannot be undone.')) {
            setCode(initialCode);
            setOutput('');
            setErrors([]);
            setTestResults([]);
        }
    };

    // Copy code to clipboard
    const copyCode = async () => {
        try {
            await navigator.clipboard.writeText(code);
            toast.success('Code copied to clipboard!');
        } catch (error) {
            toast.error('Failed to copy code');
        }
    };

    // Toggle fullscreen
    const toggleFullscreen = () => {
        setIsFullscreen(!isFullscreen);
    };

    // Render editor
    const renderEditor = () => {
        return (
            <div className="h-full relative">
                {/* Loading overlay */}
                {isLoading && (
                    <div className="absolute inset-0 bg-gray-900 bg-opacity-75 flex items-center justify-center z-10">
                        <div className="text-center">
                            <Loader className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-2" />
                            <p className="text-white">Loading Python environment...</p>
                        </div>
                    </div>
                )}

                {/* Monaco Editor */}
                <div className="h-full">
                    {typeof window !== 'undefined' && (
                        <React.Suspense fallback={<div className="h-full bg-gray-900 flex items-center justify-center">
                            <Loader className="w-6 h-6 animate-spin text-blue-400" />
                        </div>}>
                            <MonacoEditor
                                height="100%"
                                language={language}
                                value={code}
                                onChange={handleCodeChange}
                                theme={editorSettings.theme}
                                options={{
                                    fontSize: editorSettings.fontSize,
                                    wordWrap: editorSettings.wordWrap,
                                    minimap: editorSettings.minimap,
                                    readOnly,
                                    automaticLayout: true,
                                    scrollBeyondLastLine: false,
                                    renderLineHighlight: 'all',
                                    selectOnLineNumbers: true,
                                    lineNumbers: 'on',
                                    glyphMargin: true,
                                    folding: true,
                                    formatOnPaste: true,
                                    formatOnType: true
                                }}
                                onMount={(editor) => {
                                    editorRef.current = editor;
                                }}
                            />
                        </React.Suspense>
                    )}
                </div>
            </div>
        );
    };

    // Render output panel
    const renderOutput = () => {
        return (
            <div className="h-full bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
                <div className="h-full flex flex-col">
                    {/* Output Header */}
                    <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                        <div className="flex items-center space-x-2">
                            <Terminal className="w-4 h-4 text-gray-400" />
                            <span className="text-sm font-medium text-white">Output</span>
                            {executionTime > 0 && (
                                <span className="text-xs text-gray-400">
                                    ({executionTime}ms)
                                </span>
                            )}
                        </div>
                        <button
                            onClick={() => setOutput('')}
                            className="text-gray-400 hover:text-white transition-colors"
                        >
                            <X className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Output Content */}
                    <div className="flex-1 p-4 overflow-y-auto">
                        {errors.length > 0 ? (
                            <div className="space-y-2">
                                {errors.map((error, index) => (
                                    <div key={index} className="flex items-start space-x-2 text-red-400">
                                        <AlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
                                        <pre className="text-sm font-mono whitespace-pre-wrap">{error}</pre>
                                    </div>
                                ))}
                            </div>
                        ) : output ? (
                            <pre className="text-sm font-mono text-green-400 whitespace-pre-wrap">
                                {output}
                            </pre>
                        ) : (
                            <div className="text-center text-gray-500 py-8">
                                <Terminal className="w-8 h-8 mx-auto mb-2" />
                                <p>Run your code to see output here</p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    // Render test results
    const renderTests = () => {
        if (!showTests || !testCases || testCases.length === 0) {
            return (
                <div className="h-full bg-gray-900 border border-gray-700 rounded-lg flex items-center justify-center">
                    <div className="text-center text-gray-500">
                        <Target className="w-8 h-8 mx-auto mb-2" />
                        <p>No test cases available</p>
                    </div>
                </div>
            );
        }

        return (
            <div className="h-full bg-gray-900 border border-gray-700 rounded-lg overflow-hidden">
                <div className="h-full flex flex-col">
                    {/* Tests Header */}
                    <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
                        <div className="flex items-center space-x-2">
                            <Target className="w-4 h-4 text-gray-400" />
                            <span className="text-sm font-medium text-white">Test Cases</span>
                            {testResults.length > 0 && (
                                <span className="text-xs text-gray-400">
                                    ({testResults.filter(r => r.passed).length}/{testResults.length} passed)
                                </span>
                            )}
                        </div>
                        <button
                            onClick={runTests}
                            disabled={isRunning || !pyodideReady}
                            className="text-blue-400 hover:text-blue-300 transition-colors disabled:opacity-50"
                        >
                            <Play className="w-4 h-4" />
                        </button>
                    </div>

                    {/* Test Results */}
                    <div className="flex-1 overflow-y-auto">
                        {testResults.length > 0 ? (
                            <div className="space-y-2 p-4">
                                {testResults.map((result, index) => (
                                    <div
                                        key={index}
                                        className={`
                                            p-3 rounded-lg border
                                            ${result.passed 
                                                ? 'bg-green-500 bg-opacity-10 border-green-500 border-opacity-30' 
                                                : 'bg-red-500 bg-opacity-10 border-red-500 border-opacity-30'
                                            }
                                        `}
                                    >
                                        <div className="flex items-center justify-between mb-2">
                                            <span className="text-sm font-medium text-white">
                                                {result.description}
                                            </span>
                                            {result.passed ? (
                                                <CheckCircle className="w-4 h-4 text-green-400" />
                                            ) : (
                                                <X className="w-4 h-4 text-red-400" />
                                            )}
                                        </div>
                                        
                                        {result.input && (
                                            <div className="text-xs text-gray-400 mb-1">
                                                Input: {result.input}
                                            </div>
                                        )}
                                        
                                        {result.expected && (
                                            <div className="text-xs text-gray-400 mb-1">
                                                Expected: {result.expected}
                                            </div>
                                        )}
                                        
                                        {result.output && (
                                            <div className="text-xs text-gray-300">
                                                Output: {result.output}
                                            </div>
                                        )}
                                        
                                        {result.error && (
                                            <div className="text-xs text-red-400 mt-1">
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
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
            </div>
        );
    };

    // Dynamic Monaco Editor component
    const MonacoEditor = React.lazy(() => import('@monaco-editor/react'));

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

                    {/* Test Button */}
                    {showTests && testCases && testCases.length > 0 && (
                        <button
                            onClick={runTests}
                            disabled={isRunning || !pyodideReady}
                            className="flex items-center space-x-1 px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm font-medium transition-colors disabled:opacity-50"
                        >
                            <Target className="w-4 h-4" />
                            <span>Test</span>
                        </button>
                    )}

                    {/* Submit Button */}
                    {questId && (
                        <button
                            onClick={submitSolution}
                            disabled={submitSolutionMutation.isLoading}
                            className="flex items-center space-x-1 px-3 py-1 bg-purple-600 hover:bg-purple-700 text-white rounded text-sm font-medium transition-colors disabled:opacity-50"
                        >
                            <CheckCircle className="w-4 h-4" />
                            <span>Submit</span>
                        </button>
                    )}

                    {/* More Actions */}
                    <div className="flex items-center space-x-1">
                        <button
                            onClick={copyCode}
                            className="p-1 text-gray-400 hover:text-white transition-colors"
                            title="Copy Code"
                        >
                            <Copy className="w-4 h-4" />
                        </button>

                        <button
                            onClick={resetCode}
                            className="p-1 text-gray-400 hover:text-white transition-colors"
                            title="Reset Code"
                        >
                            <RotateCcw className="w-4 h-4" />
                        </button>

                        <button
                            onClick={toggleFullscreen}
                            className="p-1 text-gray-400 hover:text-white transition-colors"
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
            </div>

            {/* Main Content */}
            <div className="flex-1 flex overflow-hidden">
                {/* Code Editor */}
                <div className="flex-1 min-w-0">
                    {renderEditor()}
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
                            </button>
                        )}
                    </div>

                    {/* Tab Content */}
                    <div className="flex-1 overflow-hidden">
                        {activeTab === 'output' && renderOutput()}
                        {activeTab === 'tests' && renderTests()}
                    </div>
                </div>
            </div>

            {/* Status Bar */}
            <div className="flex items-center justify-between px-4 py-1 bg-gray-800 border-t border-gray-700 text-xs text-gray-400">
                <div className="flex items-center space-x-4">
                    <span>Python {pyodideReady ? 'Ready' : 'Loading...'}</span>
                    {executionTime > 0 && (
                        <span>Last run: {executionTime}ms</span>
                    )}
                </div>
                <div className="flex items-center space-x-2">
                    {saveCodeMutation.isLoading && (
                        <span className="flex items-center space-x-1">
                            <Loader className="w-3 h-3 animate-spin" />
                            <span>Saving...</span>
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
};

export default CodePlayground;