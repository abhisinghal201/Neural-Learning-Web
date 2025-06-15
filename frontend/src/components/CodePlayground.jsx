import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Play, 
  Square, 
  RotateCcw, 
  Save, 
  Upload, 
  Download,
  Settings,
  Maximize2,
  Minimize2,
  Terminal,
  FileText,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
  Lightbulb,
  Code,
  Eye,
  EyeOff
} from 'lucide-react';
import Editor from '@monaco-editor/react';
import { useMutation } from 'react-query';
import toast from 'react-hot-toast';
import { api } from '../utils/api';

const CodePlayground = ({ 
  quest, 
  initialCode = '', 
  onCodeSubmit, 
  onCodeChange, 
  className = '' 
}) => {
  const [code, setCode] = useState(initialCode || quest?.starter_code || '');
  const [output, setOutput] = useState('');
  const [isRunning, setIsRunning] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [activeTab, setActiveTab] = useState('editor');
  const [testResults, setTestResults] = useState(null);
  const [executionTime, setExecutionTime] = useState(0);
  const [showHints, setShowHints] = useState(false);
  const [editorSettings, setEditorSettings] = useState({
    theme: 'vs-dark',
    fontSize: 14,
    minimap: true,
    wordWrap: 'on',
    lineNumbers: 'on'
  });

  const editorRef = useRef(null);
  const pyodideRef = useRef(null);
  const startTimeRef = useRef(null);

  // Initialize Pyodide
  useEffect(() => {
    const initPyodide = async () => {
      try {
        if (!window.pyodide) {
          const pyodide = await window.loadPyodide();
          window.pyodide = pyodide;
          pyodideRef.current = pyodide;
          
          // Install common packages
          await pyodide.loadPackage(['numpy', 'matplotlib-pyodide']);
          console.log('Pyodide initialized with numpy and matplotlib');
        } else {
          pyodideRef.current = window.pyodide;
        }
      } catch (error) {
        console.error('Failed to initialize Pyodide:', error);
        toast.error('Failed to initialize Python environment');
      }
    };

    initPyodide();
  }, []);

  // Update code when quest changes
  useEffect(() => {
    if (quest?.starter_code && !initialCode) {
      setCode(quest.starter_code);
    }
  }, [quest, initialCode]);

  // Run code mutation
  const runCodeMutation = useMutation(
    async (codeToRun) => {
      if (!pyodideRef.current) {
        throw new Error('Python environment not ready');
      }

      startTimeRef.current = performance.now();
      setIsRunning(true);
      setOutput('Running code...\n');

      try {
        // Capture stdout
        pyodideRef.current.runPython(`
import sys
from io import StringIO
import contextlib

_stdout = StringIO()
_stderr = StringIO()
        `);

        // Execute user code with output capture
        const result = pyodideRef.current.runPython(`
@contextlib.contextmanager
def capture_output():
    old_stdout, old_stderr = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = _stdout, _stderr
        yield _stdout, _stderr
    finally:
        sys.stdout, sys.stderr = old_stdout, old_stderr

with capture_output() as (out, err):
    try:
${codeToRun.split('\n').map(line => '        ' + line).join('\n')}
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

stdout_value = out.getvalue()
stderr_value = err.getvalue()
        `);

        // Get captured output
        const stdout = pyodideRef.current.runPython('stdout_value');
        const stderr = pyodideRef.current.runPython('stderr_value');
        
        const endTime = performance.now();
        setExecutionTime(endTime - startTimeRef.current);

        let finalOutput = '';
        if (stdout) finalOutput += stdout;
        if (stderr) finalOutput += stderr;
        
        return finalOutput || 'Code executed successfully (no output)';
      } catch (error) {
        const endTime = performance.now();
        setExecutionTime(endTime - startTimeRef.current);
        throw new Error(`Execution error: ${error.message}`);
      }
    },
    {
      onSuccess: (result) => {
        setOutput(result);
        setIsRunning(false);
      },
      onError: (error) => {
        setOutput(`❌ ${error.message}`);
        setIsRunning(false);
        toast.error('Code execution failed');
      }
    }
  );

  // Submit quest mutation
  const submitQuestMutation = useMutation(
    async (submissionData) => {
      return api.post('/learning/quests/submit', submissionData);
    },
    {
      onSuccess: () => {
        toast.success('Quest submitted successfully!');
        if (onCodeSubmit) onCodeSubmit(code, output, testResults);
      },
      onError: (error) => {
        toast.error('Failed to submit quest: ' + error.message);
      }
    }
  );

  // Handle code execution
  const handleRunCode = () => {
    if (!code.trim()) {
      toast.error('Please write some code first');
      return;
    }
    runCodeMutation.mutate(code);
  };

  // Handle code reset
  const handleResetCode = () => {
    if (window.confirm('Are you sure you want to reset your code? This will restore the starter code.')) {
      setCode(quest?.starter_code || '');
      setOutput('');
      setTestResults(null);
    }
  };

  // Handle quest submission
  const handleSubmitQuest = () => {
    if (!quest) {
      toast.error('No quest selected');
      return;
    }

    submitQuestMutation.mutate({
      quest_id: quest.id,
      quest_title: quest.title,
      quest_type: quest.type,
      phase: quest.phase,
      week: quest.week,
      difficulty_level: quest.difficulty_level,
      code_solution: code,
      execution_result: output,
      time_to_complete_minutes: Math.round(executionTime / 1000 / 60) || 1,
      status: 'completed'
    });
  };

  // Run test cases
  const handleRunTests = async () => {
    if (!quest?.test_cases || quest.test_cases.length === 0) {
      toast.error('No test cases available for this quest');
      return;
    }

    setIsRunning(true);
    const results = [];

    try {
      for (const [index, testCase] of quest.test_cases.entries()) {
        const testCode = `
${code}

# Test case ${index + 1}
try:
    result = ${testCase.input ? JSON.stringify(testCase.input) : 'main()'}
    print(f"Test {index + 1}: {result}")
except Exception as e:
    print(f"Test {index + 1} failed: {e}")
        `;

        const result = await runCodeMutation.mutateAsync(testCode);
        
        results.push({
          testId: index + 1,
          passed: !result.includes('failed') && !result.includes('Error'),
          output: result,
          expected: testCase.expected || 'Success'
        });
      }

      setTestResults(results);
      const passedCount = results.filter(r => r.passed).length;
      
      if (passedCount === results.length) {
        toast.success(`All ${results.length} tests passed! 🎉`);
      } else {
        toast.error(`${passedCount}/${results.length} tests passed`);
      }
    } catch (error) {
      toast.error('Failed to run tests');
    } finally {
      setIsRunning(false);
    }
  };

  // Handle editor mount
  const handleEditorDidMount = (editor, monaco) => {
    editorRef.current = editor;
    
    // Set up custom themes
    monaco.editor.defineTheme('neural-dark', {
      base: 'vs-dark',
      inherit: true,
      rules: [
        { token: 'comment', foreground: '6A9955', fontStyle: 'italic' },
        { token: 'keyword', foreground: '569CD6' },
        { token: 'string', foreground: 'CE9178' },
        { token: 'number', foreground: 'B5CEA8' },
      ],
      colors: {
        'editor.background': '#0d1117',
        'editor.foreground': '#e6edf3',
        'editorLineNumber.foreground': '#7d8590',
        'editor.selectionBackground': '#264f78',
        'editor.lineHighlightBackground': '#161b22'
      }
    });

    monaco.editor.setTheme('neural-dark');

    // Add keyboard shortcuts
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.Enter, handleRunCode);
    editor.addCommand(monaco.KeyMod.CtrlCmd | monaco.KeyCode.KeyS, (e) => {
      e.preventDefault();
      // Save code locally
      localStorage.setItem(`quest_${quest?.id}_code`, code);
      toast.success('Code saved locally');
    });
  };

  // Handle code change
  const handleCodeChange = (value) => {
    setCode(value || '');
    if (onCodeChange) onCodeChange(value || '');
  };

  // Load saved code
  useEffect(() => {
    if (quest?.id) {
      const savedCode = localStorage.getItem(`quest_${quest.id}_code`);
      if (savedCode && savedCode !== quest.starter_code) {
        if (window.confirm('Found saved code for this quest. Would you like to restore it?')) {
          setCode(savedCode);
        }
      }
    }
  }, [quest]);

  return (
    <div className={`code-playground flex flex-col h-full bg-gray-900 rounded-xl overflow-hidden ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-700 bg-gray-800/50">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Code className="w-5 h-5 text-blue-400" />
            <h3 className="font-semibold text-white">
              {quest ? `${quest.title} - Code Editor` : 'Code Playground'}
            </h3>
          </div>
          
          {executionTime > 0 && (
            <div className="flex items-center gap-1 text-sm text-gray-400">
              <Clock className="w-3 h-3" />
              <span>{Math.round(executionTime)}ms</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-2">
          {quest?.hints && (
            <button
              onClick={() => setShowHints(!showHints)}
              className={`p-2 rounded-lg transition-colors ${
                showHints ? 'bg-yellow-500/20 text-yellow-400' : 'hover:bg-gray-700 text-gray-400'
              }`}
            >
              <Lightbulb className="w-4 h-4" />
            </button>
          )}
          
          <button
            onClick={() => setEditorSettings(prev => ({ ...prev, minimap: !prev.minimap }))}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400"
          >
            {editorSettings.minimap ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
          </button>
          
          <button
            onClick={() => setIsFullscreen(!isFullscreen)}
            className="p-2 hover:bg-gray-700 rounded-lg transition-colors text-gray-400"
          >
            {isFullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>

      {/* Hints Panel */}
      <AnimatePresence>
        {showHints && quest?.hints && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            className="bg-yellow-500/5 border-b border-yellow-500/20 p-4"
          >
            <div className="flex items-center gap-2 mb-2">
              <Lightbulb className="w-4 h-4 text-yellow-400" />
              <span className="font-medium text-yellow-400">Hints</span>
            </div>
            <div className="space-y-1">
              {quest.hints.map((hint, index) => (
                <div key={index} className="text-sm text-gray-300 flex items-start gap-2">
                  <span className="text-yellow-400 font-bold">{index + 1}.</span>
                  <span>{hint}</span>
                </div>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex">
        {/* Editor */}
        <div className="flex-1 flex flex-col">
          <div className="flex items-center gap-2 p-2 bg-gray-800/30 border-b border-gray-700">
            <button
              onClick={() => setActiveTab('editor')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                activeTab === 'editor' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Editor
            </button>
            <button
              onClick={() => setActiveTab('output')}
              className={`px-3 py-1 rounded text-sm transition-colors ${
                activeTab === 'output' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              Output
            </button>
            {testResults && (
              <button
                onClick={() => setActiveTab('tests')}
                className={`px-3 py-1 rounded text-sm transition-colors ${
                  activeTab === 'tests' ? 'bg-blue-500 text-white' : 'text-gray-400 hover:text-white'
                }`}
              >
                Tests ({testResults.filter(r => r.passed).length}/{testResults.length})
              </button>
            )}
          </div>

          <div className="flex-1">
            {activeTab === 'editor' && (
              <Editor
                height="100%"
                defaultLanguage="python"
                value={code}
                onChange={handleCodeChange}
                onMount={handleEditorDidMount}
                options={{
                  ...editorSettings,
                  automaticLayout: true,
                  scrollBeyondLastLine: false,
                  minimap: { enabled: editorSettings.minimap },
                  fontSize: editorSettings.fontSize,
                  wordWrap: editorSettings.wordWrap,
                  lineNumbers: editorSettings.lineNumbers,
                  renderWhitespace: 'selection',
                  bracketPairColorization: { enabled: true },
                  guides: { bracketPairs: true }
                }}
              />
            )}

            {activeTab === 'output' && (
              <div className="h-full p-4 bg-gray-900 overflow-auto">
                <pre className="text-gray-300 font-mono text-sm whitespace-pre-wrap">
                  {output || 'No output yet. Run your code to see results here.'}
                </pre>
              </div>
            )}

            {activeTab === 'tests' && testResults && (
              <div className="h-full p-4 bg-gray-900 overflow-auto space-y-3">
                {testResults.map((result, index) => (
                  <div key={index} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-white">Test {result.testId}</span>
                      <div className="flex items-center gap-1">
                        {result.passed ? (
                          <CheckCircle className="w-4 h-4 text-green-400" />
                        ) : (
                          <XCircle className="w-4 h-4 text-red-400" />
                        )}
                        <span className={`text-sm ${result.passed ? 'text-green-400' : 'text-red-400'}`}>
                          {result.passed ? 'Passed' : 'Failed'}
                        </span>
                      </div>
                    </div>
                    <pre className="text-xs text-gray-400 font-mono whitespace-pre-wrap">
                      {result.output}
                    </pre>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between p-4 border-t border-gray-700 bg-gray-800/50">
        <div className="flex items-center gap-2">
          <button
            onClick={handleRunCode}
            disabled={isRunning || !pyodideRef.current}
            className="bg-green-500 hover:bg-green-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            {isRunning ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            {isRunning ? 'Running...' : 'Run Code'}
          </button>

          {quest?.test_cases && quest.test_cases.length > 0 && (
            <button
              onClick={handleRunTests}
              disabled={isRunning}
              className="bg-blue-500 hover:bg-blue-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Zap className="w-4 h-4" />
              Run Tests
            </button>
          )}

          <button
            onClick={handleResetCode}
            className="bg-gray-600 hover:bg-gray-500 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => {
              localStorage.setItem(`quest_${quest?.id}_code`, code);
              toast.success('Code saved locally');
            }}
            className="bg-gray-600 hover:bg-gray-500 text-white px-3 py-2 rounded-lg transition-colors"
          >
            <Save className="w-4 h-4" />
          </button>

          {quest && (
            <button
              onClick={handleSubmitQuest}
              disabled={submitQuestMutation.isLoading || !output}
              className="bg-purple-500 hover:bg-purple-600 disabled:bg-gray-600 text-white px-4 py-2 rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              <Upload className="w-4 h-4" />
              Submit Quest
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodePlayground;