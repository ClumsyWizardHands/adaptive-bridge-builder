import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

const IntegrationAssistant = () => {
  const [isMinimized, setIsMinimized] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x: 20, y: 20 });
  const [activeAgents, setActiveAgents] = useState([
    { id: 1, name: 'PrincipleEngine', status: 'connected', type: 'core' },
    { id: 2, name: 'EmojiTranslator', status: 'connecting', type: 'communication' },
    { id: 3, name: 'FairnessEvaluator', status: 'disconnected', type: 'ethics' }
  ]);
  const [generatedCode, setGeneratedCode] = useState('');
  const [showCodeSnippet, setShowCodeSnippet] = useState(false);
  const dragRef = useRef(null);
  const dropZoneRef = useRef(null);

  // Assistant character - a friendly bridge/connector themed bot
  const AssistantAvatar = () => (
    <div className="relative">
      <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg">
        <svg className="w-10 h-10 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
        </svg>
      </div>
      <div className="absolute -bottom-1 -right-1 w-4 h-4 bg-green-400 rounded-full border-2 border-white animate-pulse"></div>
    </div>
  );

  const handleDragStart = (e) => {
    setIsDragging(true);
    const rect = dragRef.current.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;
    
    const handleMouseMove = (e) => {
      setPosition({
        x: e.clientX - offsetX,
        y: e.clientY - offsetY
      });
    };
    
    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
    
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  };

  const handleTestConnection = async (agentId) => {
    const agent = activeAgents.find(a => a.id === agentId);
    setActiveAgents(prev => prev.map(a => 
      a.id === agentId ? { ...a, status: 'connecting' } : a
    ));
    
    // Simulate connection test with Promise
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    setActiveAgents(prev => prev.map(a => 
      a.id === agentId ? { ...a, status: 'connected' } : a
    ));
    
    // Generate example code snippet
    const code = `# Testing connection to ${agent.name}
from adaptive_bridge_builder import BridgeConnector

bridge = BridgeConnector()
agent = bridge.connect_agent("${agent.name}", {
    "protocol": "A2A",
    "type": "${agent.type}",
    "auth": "secure_token_here"
})

# Test the connection
status = agent.test_connection()
print(f"Connection status: {status}")`;
    
    setGeneratedCode(code);
    setShowCodeSnippet(true);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const agentData = e.dataTransfer.getData('text/plain');
    
    try {
      const newAgent = JSON.parse(agentData);
      setActiveAgents(prev => [...prev, {
        id: Date.now(),
        name: newAgent.name || 'NewAgent',
        status: 'connecting',
        type: newAgent.type || 'custom'
      }]);
    } catch (error) {
      console.error('Invalid agent data:', error);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'connected': return 'bg-green-500';
      case 'connecting': return 'bg-yellow-500 animate-pulse';
      case 'disconnected': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getAgentIcon = (type) => {
    switch (type) {
      case 'core':
        return <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />;
      case 'communication':
        return <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />;
      case 'ethics':
        return <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 6l3 1m0 0l-3 9a5.002 5.002 0 006.001 0M6 7l3 9M6 7l6-2m6 2l3-1m-3 1l-3 9a5.002 5.002 0 006.001 0M18 7l3 9m-3-9l-6-2m0-2v2m0 16V5m0 16H9m3 0h3" />;
      default:
        return <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />;
    }
  };

  return (
    <motion.div
      ref={dragRef}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      style={{
        position: 'fixed',
        left: `${position.x}px`,
        top: `${position.y}px`,
        zIndex: 9999
      }}
      className={`${isDragging ? 'cursor-move' : ''}`}
    >
      <AnimatePresence>
        {!isMinimized ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
            className="bg-white rounded-2xl shadow-2xl border border-gray-200 overflow-hidden"
            style={{ width: '400px', maxHeight: '600px' }}
          >
            {/* Header */}
            <div 
              className="bg-gradient-to-r from-blue-500 to-purple-600 p-4 cursor-move flex items-center justify-between"
              onMouseDown={handleDragStart}
            >
              <div className="flex items-center space-x-3">
                <AssistantAvatar />
                <div>
                  <h3 className="text-white font-bold text-lg">Bridge Assistant</h3>
                  <p className="text-blue-100 text-sm">Agent Integration Helper</p>
                </div>
              </div>
              <button
                onClick={() => setIsMinimized(true)}
                className="text-white hover:bg-white/20 p-2 rounded-lg transition-colors"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </div>

            {/* Content */}
            <div className="p-4 space-y-4 overflow-y-auto" style={{ maxHeight: '500px' }}>
              {/* Connection Status */}
              <div>
                <h4 className="font-semibold text-gray-700 mb-2">Active Agents</h4>
                <div className="space-y-2">
                  {activeAgents.map((agent) => (
                    <motion.div
                      key={agent.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      className="bg-gray-50 rounded-lg p-3 flex items-center justify-between"
                    >
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
                        <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          {getAgentIcon(agent.type)}
                        </svg>
                        <div>
                          <p className="font-medium text-gray-800">{agent.name}</p>
                          <p className="text-xs text-gray-500 capitalize">{agent.type} • {agent.status}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => handleTestConnection(agent.id)}
                        className="text-blue-600 hover:bg-blue-50 px-3 py-1 rounded-md text-sm font-medium transition-colors"
                      >
                        Test
                      </button>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* Drag and Drop Zone */}
              <div
                ref={dropZoneRef}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-blue-400 transition-colors"
              >
                <svg className="w-12 h-12 text-gray-400 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                <p className="text-gray-600 font-medium">Drop agent configuration here</p>
                <p className="text-gray-500 text-sm mt-1">or browse to add manually</p>
              </div>

              {/* Code Snippet */}
              <AnimatePresence>
                {showCodeSnippet && generatedCode && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="bg-gray-900 rounded-lg p-4 relative"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-white font-medium text-sm">Generated Code</h4>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(generatedCode);
                        }}
                        className="text-gray-400 hover:text-white text-sm"
                      >
                        Copy
                      </button>
                    </div>
                    <pre className="text-green-400 text-xs overflow-x-auto">
                      <code>{generatedCode}</code>
                    </pre>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Quick Actions */}
              <div className="grid grid-cols-2 gap-2">
                <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors font-medium">
                  Test All Connections
                </button>
                <button className="bg-purple-500 text-white px-4 py-2 rounded-lg hover:bg-purple-600 transition-colors font-medium">
                  Generate Bridge Code
                </button>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className="cursor-pointer"
            onClick={() => setIsMinimized(false)}
          >
            <AssistantAvatar />
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default IntegrationAssistant;
