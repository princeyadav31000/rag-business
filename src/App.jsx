import React, { useState, useEffect, useRef } from "react";
import {
  Send,
  Bot,
  User,
  Globe,
  ExternalLink,
  Clock,
  Database,
  Zap,
  CheckCircle,
  AlertCircle,
  BarChart3,
  Minimize2,
  Maximize2,
  FileUp,
  X as LucideX,
} from "lucide-react";

const RAGChatbot = () => {
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isConnected, setIsConnected] = useState(false);
  const [stats, setStats] = useState(null);
  const [showStats, setShowStats] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [pdfStatus, setPdfStatus] = useState("");
  const [searchMode, setSearchMode] = useState("both"); // 'rag', 'pdf', 'both'

  const API_BASE_URL = "https://4289-2a02-4780-12-cc70-00-1.ngrok-free.app";
  // const API_BASE_URL = "http://localhost:5000";

  useEffect(() => {
    checkServerHealth();
    loadStats();
    // Add welcome message
    setMessages([
      {
        id: 1,
        type: "bot",
        content:
          "👋 Hello! I'm your RAG Assistant. I can help you find information from your knowledge base. What would you like to know?",
        timestamp: new Date().toISOString(),
      },
    ]);
    // Automatically clear PDF data on page load
    fetch(`${API_BASE_URL}/clear-pdf`, {
      method: "POST",
      headers: {
        "ngrok-skip-browser-warning": "true",
      },
    });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const checkServerHealth = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/health`, {
        method: "GET",
        headers: {
          "ngrok-skip-browser-warning": "true", // Skip ngrok warning
        }
      });
      const data = await response.json();
      setIsConnected(data.status === "healthy" && data.model_loaded);
    } catch (error) {
      setIsConnected(false);
    }
  };

  const loadStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/stats`, {
        method: "GET",
        headers: {
          "ngrok-skip-browser-warning": "true", // Skip ngrok warning
        }
      });
      const data = await response.json();
      setStats(data);
    } catch (error) {
      console.error("Failed to load stats:", error);
    }
  };

  const handlePdfChange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setPdfFile(file);
    setPdfStatus("Uploading...");
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE_URL}/upload-pdf`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
        body: formData,
      });
      const data = await response.json();
      if (response.ok) {
        setPdfStatus(`Uploaded: ${data.filename} (${data.chunks_created} chunks)`);
      } else {
        setPdfStatus(`❌ ${data.error || "Failed to upload PDF"}`);
        setPdfFile(null);
      }
    } catch (error) {
      setPdfStatus(`❌ Error: ${error.message}`);
      setPdfFile(null);
    }
  };

  const handleClearPdf = async () => {
    setPdfStatus("Clearing...");
    try {
      const response = await fetch(`${API_BASE_URL}/clear-pdf`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true",
        },
      });
      const data = await response.json();
      if (response.ok) {
        setPdfFile(null);
        setPdfStatus("PDF cleared");
      } else {
        setPdfStatus(`❌ ${data.error || "Failed to clear PDF"}`);
      }
    } catch (error) {
      setPdfStatus(`❌ Error: ${error.message}`);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      id: Date.now(),
      type: "user",
      content: inputMessage,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputMessage("");
    setIsLoading(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "ngrok-skip-browser-warning": "true",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message: inputMessage, search_mode: searchMode }),
      });

      const data = await response.json();

      if (response.ok) {
        const botMessage = {
          id: Date.now() + 1,
          type: "bot",
          content: data.message,
          source: data.source,
          score: data.score,
          link: data.link,
          type_info: data.type,
          areas: data.areas,
          timestamp: data.timestamp,
        };
        setMessages((prev) => [...prev, botMessage]);
      } else {
        throw new Error(data.error || "Failed to get response");
      }
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        type: "bot",
        content: `❌ Sorry, I encountered an error: ${error.message}`,
        timestamp: new Date().toISOString(),
        isError: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (content) => {
    // Convert markdown-style bold to HTML
    return content.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  };

  const ConnectionStatus = () => (
    <div
      className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all ${
        isConnected
          ? "bg-green-50 text-green-700 border border-green-200"
          : "bg-red-50 text-red-700 border border-red-200"
      }`}
    >
      {isConnected ? (
        <CheckCircle className="w-4 h-4" />
      ) : (
        <AlertCircle className="w-4 h-4" />
      )}
      {isConnected ? "Connected" : "Disconnected"}
    </div>
  );

  const StatsPanel = () => (
    <div
      className={`fixed top-20 right-4 w-80 bg-white rounded-xl shadow-xl border border-gray-200 transition-all duration-300 z-10 ${
        showStats
          ? "translate-x-0 opacity-100"
          : "translate-x-full opacity-0 pointer-events-none"
      }`}
    >
      <div className="p-4 border-b border-gray-100 flex items-center justify-between">
        <h3 className="font-semibold text-gray-900 flex items-center gap-2">
          <Database className="w-4 h-4 text-blue-600" />
          Knowledge Base
        </h3>
        <button
          onClick={() => setShowStats(false)}
          className="p-1 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <Minimize2 className="w-4 h-4 text-gray-500" />
        </button>
      </div>

      {stats && (
        <div className="p-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-blue-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-blue-600">
                {stats.total_documents}
              </div>
              <div className="text-sm text-gray-600">Total Documents</div>
            </div>
            <div className="bg-green-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-green-600">
                {stats.documents_with_links}
              </div>
              <div className="text-sm text-gray-600">With Links</div>
            </div>
            <div className="bg-purple-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-purple-600">
                {stats.unique_types?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Document Types</div>
            </div>
            <div className="bg-orange-50 rounded-lg p-3">
              <div className="text-2xl font-bold text-orange-600">
                {stats.unique_areas?.length || 0}
              </div>
              <div className="text-sm text-gray-600">Topic Areas</div>
            </div>
          </div>

          {stats.unique_types && stats.unique_types.length > 0 && (
            <div className="mt-4">
              <h4 className="font-medium text-gray-700 mb-2">Document Types</h4>
              <div className="flex flex-wrap gap-1">
                {stats.unique_types.slice(0, 6).map((type, index) => (
                  <span
                    key={index}
                    className="px-2 py-1 bg-gray-100 text-gray-700 text-xs rounded-full"
                  >
                    {type}
                  </span>
                ))}
                {stats.unique_types.length > 6 && (
                  <span className="px-2 py-1 bg-gray-100 text-gray-500 text-xs rounded-full">
                    +{stats.unique_types.length - 6} more
                  </span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 sticky top-0 z-20">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="bg-blue-600 p-2 rounded-xl">
                <Bot className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">
                  RAG Assistant
                </h1>
                <p className="text-sm text-gray-600">
                  Intelligent Knowledge Base Assistant
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <ConnectionStatus />
              <button
                onClick={() => setShowStats(!showStats)}
                className="flex items-center gap-2 px-3 py-1.5 bg-gray-100 hover:bg-gray-200 text-gray-700 rounded-lg transition-colors"
              >
                <BarChart3 className="w-4 h-4" />
                Stats
              </button>
            </div>
          </div>
        </div>
      </header>

      <StatsPanel />

      {/* Search Mode Selection - Top Right */}
      <div className="max-w-4xl mx-auto px-4 pt-4 flex justify-end">
        <div className="flex items-center gap-2 bg-white border border-gray-200 rounded-xl shadow-sm px-3 py-2">
          <span className="text-xs text-gray-500 mr-2 font-medium">Search Mode:</span>
          <button
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors border ${searchMode === "rag" ? "bg-blue-600 text-white border-blue-600" : "bg-gray-50 text-gray-700 border-gray-200 hover:bg-blue-50"}`}
            onClick={() => setSearchMode("rag")}
            disabled={isLoading}
          >
            RAG Only
          </button>
          <button
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors border ${searchMode === "pdf" ? "bg-blue-600 text-white border-blue-600" : "bg-gray-50 text-gray-700 border-gray-200 hover:bg-blue-50"}`}
            onClick={() => setSearchMode("pdf")}
            disabled={isLoading}
          >
            PDF Only
          </button>
          <button
            className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors border ${searchMode === "both" ? "bg-blue-600 text-white border-blue-600" : "bg-gray-50 text-gray-700 border-gray-200 hover:bg-blue-50"}`}
            onClick={() => setSearchMode("both")}
            disabled={isLoading}
          >
            Both
          </button>
        </div>
      </div>

      {/* Main Chat Container */}
      <div className="max-w-4xl mx-auto px-4 py-6 h-[calc(100vh-88px)] flex flex-col">
        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto bg-white rounded-xl border border-gray-200 shadow-sm">
          <div className="p-6 space-y-6">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex gap-4 ${
                  message.type === "user" ? "flex-row-reverse" : ""
                }`}
              >
                {/* Avatar */}
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                    message.type === "user" ? "bg-blue-600" : "bg-gray-100"
                  }`}
                >
                  {message.type === "user" ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-gray-600" />
                  )}
                </div>

                {/* Message Content */}
                <div
                  className={`flex-1 max-w-3xl ${
                    message.type === "user" ? "text-right" : ""
                  }`}
                >
                  <div
                    className={`inline-block p-4 rounded-2xl ${
                      message.type === "user"
                        ? "bg-blue-600 text-white"
                        : message.isError
                        ? "bg-red-50 text-red-800 border border-red-200"
                        : "bg-gray-50 text-gray-900"
                    } ${
                      message.type === "user"
                        ? "rounded-br-md"
                        : "rounded-bl-md"
                    }`}
                  >
                    <div
                      dangerouslySetInnerHTML={{
                        __html: formatMessage(message.content),
                      }}
                      className={`prose prose-sm max-w-none ${
                        message.type === "user" ? "prose-invert" : ""
                      }`}
                    />

                    {/* Source Information */}
                    {message.source && (
                      <div className="mt-3 pt-3 border-t border-gray-200">
                        <div className="flex items-center gap-2 text-xs text-gray-600">
                          <Globe className="w-3 h-3" />
                          <span className="font-medium">Source:</span>
                          <span>{message.source}</span>
                          {message.link && (
                            <a
                              href={message.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-blue-600 hover:text-blue-800 flex items-center gap-1 ml-2"
                            >
                              <ExternalLink className="w-3 h-3" />
                              View
                            </a>
                          )}
                        </div>
                        {message.score >= 0 && (
                          <div className="flex items-center gap-2 text-xs text-gray-600 mt-1">
                            <Zap className="w-3 h-3" />
                            <span>
                              Relevance: {(message.score * 100).toFixed(1)}%
                            </span>
                          </div>
                        )}
                        {message.areas && message.areas.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-2">
                            {message.areas.map((area, index) => (
                              <span
                                key={index}
                                className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full"
                              >
                                {area}
                              </span>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  <div
                    className={`text-xs text-gray-500 mt-2 flex items-center gap-1 ${
                      message.type === "user" ? "justify-end" : "justify-start"
                    }`}
                  >
                    <Clock className="w-3 h-3" />
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}

            {/* Loading Animation */}
            {isLoading && (
              <div className="flex gap-4">
                <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-gray-600" />
                </div>
                <div className="flex-1">
                  <div className="inline-block bg-gray-50 rounded-2xl rounded-bl-md p-4">
                    <div className="flex items-center gap-2">
                      <div className="flex space-x-1">
                        <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.1s" }}
                        ></div>
                        <div
                          className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                          style={{ animationDelay: "0.2s" }}
                        ></div>
                      </div>
                      <span className="text-gray-600 text-sm ml-2">
                        Thinking...
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="mt-4 bg-white rounded-xl border border-gray-200 shadow-sm p-3">
          <div className="flex gap-2 items-end relative">
            {/* PDF Upload Button */}
            <div className="flex items-center">
              <label htmlFor="pdf-upload" className="cursor-pointer flex items-center justify-center w-10 h-10 rounded-lg bg-gray-100 hover:bg-blue-100 border border-gray-200 transition-colors" title="Upload PDF">
                <FileUp className="w-5 h-5 text-blue-600" />
                <input
                  id="pdf-upload"
                  type="file"
                  accept="application/pdf"
                  onChange={handlePdfChange}
                  disabled={isLoading}
                  className="hidden"
                />
              </label>
            </div>
            {/* PDF Filename and Clear */}
            {pdfFile && (
              <div className="flex items-center gap-1 bg-blue-50 text-blue-700 px-2 py-1 rounded-lg text-xs ml-1">
                <span className="truncate max-w-[120px]">{pdfFile.name}</span>
                <button onClick={handleClearPdf} className="ml-1 hover:text-red-600" title="Clear PDF" disabled={isLoading}>
                  <LucideX className="w-4 h-4" />
                </button>
              </div>
            )}
            {/* PDF Status */}
            {pdfStatus && !pdfFile && (
              <div className="text-xs text-gray-500 ml-2">{pdfStatus}</div>
            )}
            {/* Chat Input */}
            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  isConnected
                    ? "Ask me anything about your knowledge base..."
                    : "Please check server connection..."
                }
                disabled={!isConnected || isLoading}
                className="w-full resize-none bg-gray-50 text-gray-900 rounded-xl px-4 py-3 pr-12 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:bg-white transition-all duration-200 placeholder-gray-500 disabled:opacity-50 disabled:cursor-not-allowed border border-gray-200"
                rows={1}
                style={{
                  minHeight: "48px",
                  maxHeight: "120px",
                }}
                onInput={(e) => {
                  e.target.style.height = "auto";
                  e.target.style.height =
                    Math.min(e.target.scrollHeight, 120) + "px";
                }}
              />
              <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                <kbd className="px-2 py-1 text-xs font-medium text-gray-500 bg-gray-100 border border-gray-200 rounded">
                  Enter
                </kbd>
              </div>
            </div>
            {/* Send Button */}
            <button
              onClick={sendMessage}
              disabled={!inputMessage.trim() || isLoading || !isConnected}
              className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white p-3 rounded-xl transition-all duration-200 disabled:cursor-not-allowed shadow-sm hover:shadow-md active:transform active:scale-95 ml-1"
              title="Send"
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center justify-between mt-2 text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <span>💡 Be specific in your questions for better results</span>
            </div>
            <div className="flex items-center gap-2">
              {!isConnected && (
                <span className="text-red-500 flex items-center gap-1">
                  <AlertCircle className="w-3 h-3" />
                  Server disconnected
                </span>
              )}
              {isLoading && (
                <span className="text-blue-500 flex items-center gap-1">
                  <div className="w-3 h-3 border border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                  Processing...
                </span>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default RAGChatbot;
