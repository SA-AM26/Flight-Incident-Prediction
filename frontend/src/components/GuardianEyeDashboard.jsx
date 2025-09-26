// frontend/src/components/GuardianEyeDashboard.jsx

import React, { useState, useEffect, useRef } from "react";
import * as THREE from "three";
import {
  Plane,
  AlertTriangle,
  Eye,
  Globe,
  Radar,
  Gauge,
  Navigation,
  Activity,
} from "lucide-react";

const GuardianEyeDashboard = () => {
  const [selectedAirline, setSelectedAirline] = useState("All Airlines");
  const [selectedTailNumber, setSelectedTailNumber] = useState("All Aircraft");
  const [selectedAircraftType, setSelectedAircraftType] = useState("All Types");
  const [currentTime, setCurrentTime] = useState(new Date());
  const [alertLevel, setAlertLevel] = useState("NORMAL");

  const globeRef = useRef();
  const sceneRef = useRef();
  const rendererRef = useRef();
  const animationRef = useRef();

  // Airlines list
  const airlines = [
    "All Airlines",
    "Air India (AI)",
    "IndiGo (6E)",
    "SpiceJet (SG)",
    "Vistara (UK)",
    "GoFirst (G8)",
    "AirAsia India (I5)",
  ];

  // Aircraft types
  const aircraftTypes = [
    "All Types",
    "Boeing 737-800",
    "Boeing 737 MAX 8",
    "Airbus A320neo",
    "Airbus A321",
    "ATR 72-600",
    "Bombardier Q400",
    "Boeing 787-8",
    "Airbus A350-900",
  ];

  // Dummy aircraft data generator
  const generateAircraftData = () => {
    const tailNumbers = [];
    const aircraftData = [];
    const prefixes = ["VT-", "A6-", "9V-", "HS-"];
    const suffixes = ["ABC", "DEF", "GHI", "JKL", "MNO", "PQR", "STU", "VWX", "YZA"];

    for (let i = 0; i < 50; i++) {
      const prefix = prefixes[Math.floor(Math.random() * prefixes.length)];
      const suffix = suffixes[Math.floor(Math.random() * suffixes.length)];
      const tailNumber = `${prefix}${suffix}`;
      tailNumbers.push(tailNumber);

      const airline =
        airlines[Math.floor(Math.random() * (airlines.length - 1)) + 1];
      const aircraftType =
        aircraftTypes[Math.floor(Math.random() * (aircraftTypes.length - 1)) + 1];

      // Risk factors
      const engineHealth = Math.random() * 100;
      const structuralIntegrity = Math.random() * 100;
      const avionicsStatus = Math.random() * 100;
      const maintenanceScore = Math.random() * 100;
      const pilotExperience = Math.random() * 100;
      const weatherImpact = Math.random() * 100;

      const overallRisk =
        (100 - engineHealth) * 0.25 +
        (100 - structuralIntegrity) * 0.2 +
        (100 - avionicsStatus) * 0.15 +
        (100 - maintenanceScore) * 0.2 +
        (100 - pilotExperience) * 0.1 +
        weatherImpact * 0.1;

      let riskLevel = "LOW";
      let riskColor = "#10B981";
      if (overallRisk > 70) {
        riskLevel = "CRITICAL";
        riskColor = "#DC2626";
      } else if (overallRisk > 50) {
        riskLevel = "HIGH";
        riskColor = "#F59E0B";
      } else if (overallRisk > 30) {
        riskLevel = "MEDIUM";
        riskColor = "#3B82F6";
      }

      aircraftData.push({
        tailNumber,
        airline: airline.split(" (")[0],
        aircraftType,
        status: Math.random() > 0.3 ? "IN-FLIGHT" : "GROUNDED",
        altitude: Math.floor(Math.random() * 40000 + 5000),
        speed: Math.floor(Math.random() * 500 + 200),
        heading: Math.floor(Math.random() * 360),
        lat: (Math.random() - 0.5) * 40 + 20,
        lng: (Math.random() - 0.5) * 60 + 77,
        engineHealth,
        structuralIntegrity,
        avionicsStatus,
        maintenanceScore,
        pilotExperience,
        weatherImpact,
        overallRisk,
        riskLevel,
        riskColor,
        flightHours: Math.floor(Math.random() * 50000),
        lastMaintenance: Math.floor(Math.random() * 30),
        nextMaintenance: Math.floor(Math.random() * 100 + 50),
      });
    }
    return { tailNumbers: ["All Aircraft", ...tailNumbers], aircraftData };
  };

  const { tailNumbers, aircraftData } = generateAircraftData();

  // Filtering logic
  const filteredAircraft = aircraftData.filter((aircraft) => {
    if (selectedAirline !== "All Airlines" && !selectedAirline.includes(aircraft.airline)) return false;
    if (selectedTailNumber !== "All Aircraft" && aircraft.tailNumber !== selectedTailNumber) return false;
    if (selectedAircraftType !== "All Types" && aircraft.aircraftType !== selectedAircraftType) return false;
    return true;
  });

  const availableTailNumbers = ["All Aircraft", ...aircraftData
    .filter((aircraft) => {
      if (selectedAirline !== "All Airlines" && !selectedAirline.includes(aircraft.airline)) return false;
      if (selectedAircraftType !== "All Types" && aircraft.aircraftType !== selectedAircraftType) return false;
      return true;
    })
    .map((aircraft) => aircraft.tailNumber)];

  // 3D globe effect
  useEffect(() => {
    if (!globeRef.current) return;
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, 400 / 300, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(400, 300);
    renderer.setClearColor(0x000000, 0);
    globeRef.current.appendChild(renderer.domElement);

    // Earth
    const geometry = new THREE.SphereGeometry(2, 32, 32);
    const material = new THREE.MeshBasicMaterial({
      color: 0x2563eb,
      wireframe: true,
      transparent: true,
      opacity: 0.3,
    });
    const earth = new THREE.Mesh(geometry, material);
    scene.add(earth);

    // Aircraft markers
    filteredAircraft.forEach((aircraft) => {
      const phi = (90 - aircraft.lat) * (Math.PI / 180);
      const theta = (aircraft.lng + 180) * (Math.PI / 180);
      const x = 2.1 * Math.sin(phi) * Math.cos(theta);
      const y = 2.1 * Math.cos(phi);
      const z = 2.1 * Math.sin(phi) * Math.sin(theta);
      const aircraftGeometry = new THREE.BoxGeometry(0.1, 0.02, 0.1);
      const aircraftMaterial = new THREE.MeshBasicMaterial({
        color:
          aircraft.riskLevel === "CRITICAL"
            ? 0xff0000
            : aircraft.riskLevel === "HIGH"
            ? 0xffaa00
            : aircraft.riskLevel === "MEDIUM"
            ? 0x0088ff
            : 0x00ff00,
      });
      const aircraftMesh = new THREE.Mesh(aircraftGeometry, aircraftMaterial);
      aircraftMesh.position.set(x, y, z);
      scene.add(aircraftMesh);
    });

    camera.position.z = 5;
    sceneRef.current = scene;
    rendererRef.current = renderer;

    const animate = () => {
      animationRef.current = requestAnimationFrame(animate);
      earth.rotation.y += 0.005;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      if (globeRef.current && renderer.domElement) {
        globeRef.current.removeChild(renderer.domElement);
      }
    };
  }, [filteredAircraft]);

  // Time updates
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Alert level
  useEffect(() => {
    const criticalCount = filteredAircraft.filter((a) => a.riskLevel === "CRITICAL").length;
    const highCount = filteredAircraft.filter((a) => a.riskLevel === "HIGH").length;
    if (criticalCount > 0) setAlertLevel("CRITICAL");
    else if (highCount > 2) setAlertLevel("HIGH");
    else if (highCount > 0) setAlertLevel("ELEVATED");
    else setAlertLevel("NORMAL");
  }, [filteredAircraft]);

  // Pick selected aircraft
  const selectedAircraftData =
    selectedTailNumber !== "All Aircraft"
      ? aircraftData.find((a) => a.tailNumber === selectedTailNumber)
      : null;

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black text-white overflow-hidden">
      {/* Header */}
      <div className="bg-black bg-opacity-50 backdrop-blur-sm border-b border-blue-500/30 p-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Eye className="w-8 h-8 text-blue-400" />
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
            GUARDIAN EYE
          </h1>
          <span className="text-sm text-gray-400">Aviation Ops Center</span>
        </div>
        <div className="flex items-center space-x-6">
          <div className="text-right">
            <div className="text-lg font-mono">{currentTime.toLocaleTimeString()}</div>
            <div className="text-sm text-gray-400">{currentTime.toLocaleDateString()}</div>
          </div>
          <div
            className={`px-4 py-2 rounded-lg font-bold ${
              alertLevel === "CRITICAL"
                ? "bg-red-600 animate-pulse"
                : alertLevel === "HIGH"
                ? "bg-orange-600"
                : alertLevel === "ELEVATED"
                ? "bg-yellow-600 text-black"
                : "bg-green-600"
            }`}
          >
            ALERT: {alertLevel}
          </div>
        </div>
      </div>

      {/* Layout */}
      <div className="grid grid-cols-12 gap-4 p-4 h-screen">
        {/* Sidebar */}
        <div className="col-span-3 space-y-4">
          <div className="bg-black bg-opacity-40 rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Navigation className="w-5 h-5 mr-2 text-blue-400" />
              Aircraft Selection
            </h3>
            <div className="space-y-3">
              {/* Airline filter */}
              <select
                value={selectedAirline}
                onChange={(e) => {
                  setSelectedAirline(e.target.value);
                  setSelectedTailNumber("All Aircraft");
                }}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {airlines.map((airline) => (
                  <option key={airline} value={airline}>
                    {airline}
                  </option>
                ))}
              </select>
              {/* Type filter */}
              <select
                value={selectedAircraftType}
                onChange={(e) => {
                  setSelectedAircraftType(e.target.value);
                  setSelectedTailNumber("All Aircraft");
                }}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {aircraftTypes.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
              {/* Tail filter */}
              <select
                value={selectedTailNumber}
                onChange={(e) => setSelectedTailNumber(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {availableTailNumbers.map((tail) => (
                  <option key={tail} value={tail}>
                    {tail}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Fleet Overview */}
          <div className="bg-black bg-opacity-40 rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Radar className="w-5 h-5 mr-2 text-green-400" />
              Fleet Overview
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-blue-400">{filteredAircraft.length}</div>
                <div className="text-xs text-gray-400">Total Aircraft</div>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-green-400">
                  {filteredAircraft.filter((a) => a.status === "IN-FLIGHT").length}
                </div>
                <div className="text-xs text-gray-400">In Flight</div>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-red-400">
                  {filteredAircraft.filter((a) => a.riskLevel === "CRITICAL").length}
                </div>
                <div className="text-xs text-gray-400">Critical Risk</div>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-orange-400">
                  {filteredAircraft.filter((a) => a.riskLevel === "HIGH").length}
                </div>
                <div className="text-xs text-gray-400">High Risk</div>
              </div>
            </div>
          </div>
        </div>

        {/* Globe + List */}
        <div className="col-span-6 space-y-4">
          <div className="bg-black bg-opacity-40 rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Globe className="w-5 h-5 mr-2 text-blue-400" />
              Global Aircraft Tracking
            </h3>
            <div ref={globeRef} className="w-full flex justify-center"></div>
          </div>
          <div className="bg-black bg-opacity-40 rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-green-400" />
              Active Aircraft Monitor
            </h3>
            <div className="max-h-64 overflow-y-auto space-y-2">
