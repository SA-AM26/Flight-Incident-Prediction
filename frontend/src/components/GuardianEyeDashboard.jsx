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
  const [aircraftData, setAircraftData] = useState([]);
  const [selectedAirline, setSelectedAirline] = useState("All Airlines");
  const [selectedTailNumber, setSelectedTailNumber] = useState("All Aircraft");
  const [selectedAircraftType, setSelectedAircraftType] = useState("All Types");
  const [currentTime, setCurrentTime] = useState(new Date());
  const [alertLevel, setAlertLevel] = useState("NORMAL");

  const globeRef = useRef();
  const sceneRef = useRef();
  const rendererRef = useRef();
  const animationRef = useRef();

  // ✅ Replace this with your deployed Streamlit Cloud backend URL
  const BACKEND_URL = "https://your-backend.streamlit.app/?data=flights";

  // ------------------------------------------------------------------
  // Fetch real backend data
  // ------------------------------------------------------------------
  useEffect(() => {
    fetch(BACKEND_URL)
      .then((res) => res.json())
      .then((data) => {
        // Normalize backend dataframe rows → frontend format
        const mapped = data.map((row) => ({
          tailNumber: row.tail_number,
          airline: row.airline,
          aircraftType: row.aircraft_type,
          status: row.status,
          altitude: row.altitude,
          speed: row.speed,
          heading: row.heading,
          lat: row.current_lat,
          lng: row.current_lng,
          engineHealth: row.engine_health,
          structuralIntegrity: row.structural_integrity,
          avionicsStatus: row.avionics_status,
          maintenanceScore: row.maintenance_score,
          pilotExperience: row.pilot_experience,
          weatherImpact: row.weather_score * 100, // scale to %
          overallRisk: row.incident_probability * 100,
          riskLevel: row.risk_level,
          flightHours: row.flight_hours,
          lastMaintenance: row.last_maintenance_days,
          nextMaintenance: Math.floor(Math.random() * 200 + 50), // backend doesn't have → fake
        }));

        setAircraftData(mapped);
      })
      .catch((err) => {
        console.error("❌ Failed to fetch flights:", err);
      });
  }, []);

  // ------------------------------------------------------------------
  // Filters
  // ------------------------------------------------------------------
  const filteredAircraft = aircraftData.filter((aircraft) => {
    if (
      selectedAirline !== "All Airlines" &&
      aircraft.airline !== selectedAirline
    )
      return false;
    if (
      selectedTailNumber !== "All Aircraft" &&
      aircraft.tailNumber !== selectedTailNumber
    )
      return false;
    if (
      selectedAircraftType !== "All Types" &&
      aircraft.aircraftType !== selectedAircraftType
    )
      return false;
    return true;
  });

  const availableTailNumbers = [
    "All Aircraft",
    ...aircraftData
      .filter((aircraft) => {
        if (
          selectedAirline !== "All Airlines" &&
          aircraft.airline !== selectedAirline
        )
          return false;
        if (
          selectedAircraftType !== "All Types" &&
          aircraft.aircraftType !== selectedAircraftType
        )
          return false;
        return true;
      })
      .map((aircraft) => aircraft.tailNumber),
  ];

  // ------------------------------------------------------------------
  // Globe rendering (Three.js)
  // ------------------------------------------------------------------
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
      const color =
        aircraft.riskLevel === "CRITICAL"
          ? 0xff0000
          : aircraft.riskLevel === "HIGH"
          ? 0xffaa00
          : aircraft.riskLevel === "MEDIUM"
          ? 0x0088ff
          : 0x00ff00;
      const aircraftMaterial = new THREE.MeshBasicMaterial({ color });
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
      if (globeRef.current && renderer.domElement)
        globeRef.current.removeChild(renderer.domElement);
    };
  }, [filteredAircraft]);

  // ------------------------------------------------------------------
  // Timers + alerting
  // ------------------------------------------------------------------
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const criticalCount = filteredAircraft.filter(
      (a) => a.riskLevel === "CRITICAL"
    ).length;
    const highCount = filteredAircraft.filter(
      (a) => a.riskLevel === "HIGH"
    ).length;

    if (criticalCount > 0) setAlertLevel("CRITICAL");
    else if (highCount > 2) setAlertLevel("HIGH");
    else if (highCount > 0) setAlertLevel("ELEVATED");
    else setAlertLevel("NORMAL");
  }, [filteredAircraft]);

  const selectedAircraftData =
    selectedTailNumber !== "All Aircraft"
      ? aircraftData.find((a) => a.tailNumber === selectedTailNumber)
      : null;

  // ------------------------------------------------------------------
  // UI
  // ------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-blue-900 to-black text-white overflow-hidden">
      {/* HEADER */}
      <div className="bg-black bg-opacity-50 backdrop-blur-sm border-b border-blue-500/30 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Eye className="w-8 h-8 text-blue-400" />
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              GUARDIAN EYE
            </h1>
            <span className="text-sm text-gray-400">
              Aviation Operations Center
            </span>
          </div>
          <div className="flex items-center space-x-6">
            <div className="text-right">
              <div className="text-lg font-mono">
                {currentTime.toLocaleTimeString()}
              </div>
              <div className="text-sm text-gray-400">
                {currentTime.toLocaleDateString()}
              </div>
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
      </div>

      {/* BODY */}
      <div className="grid grid-cols-12 gap-4 p-4 h-screen">
        {/* SIDEBAR */}
        <div className="col-span-3 space-y-4">
          {/* Filters */}
          <div className="bg-black bg-opacity-40 backdrop-blur-sm rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Navigation className="w-5 h-5 mr-2 text-blue-400" />
              Aircraft Selection
            </h3>
            <div className="space-y-3">
              <select
                value={selectedAirline}
                onChange={(e) => {
                  setSelectedAirline(e.target.value);
                  setSelectedTailNumber("All Aircraft");
                }}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {["All Airlines", ...new Set(aircraftData.map((a) => a.airline))].map(
                  (a) => (
                    <option key={a}>{a}</option>
                  )
                )}
              </select>
              <select
                value={selectedAircraftType}
                onChange={(e) => {
                  setSelectedAircraftType(e.target.value);
                  setSelectedTailNumber("All Aircraft");
                }}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {["All Types", ...new Set(aircraftData.map((a) => a.aircraftType))].map(
                  (t) => (
                    <option key={t}>{t}</option>
                  )
                )}
              </select>
              <select
                value={selectedTailNumber}
                onChange={(e) => setSelectedTailNumber(e.target.value)}
                className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-white"
              >
                {availableTailNumbers.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          {/* Fleet summary */}
          <div className="bg-black bg-opacity-40 backdrop-blur-sm rounded-lg border border-blue-500/30 p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Radar className="w-5 h-5 mr-2 text-green-400" />
              Fleet Overview
            </h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-blue-400">
                  {filteredAircraft.length}
                </div>
                <div className="text-xs text-gray-400">Total</div>
              </div>
              <div className="bg-gray-800 rounded p-3 text-center">
                <div className="text-2xl font-bold text-green-400">
                  {filteredAircraft.filter((a) => a.status === "IN-FLIGHT").length}
                </div>
                <div className="text-xs text-gray-400">In Flight</div>
              </div>
            </div>
          </div>
        </div>

        {/* GLOBE + TABLE */}
        <div className="col-span-6 space-y-4">
          <div className="bg-black bg-opacity-40 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Globe className="w-5 h-5 mr-2 text-blue-400" />
              Global Aircraft Tracking
            </h3>
            <div ref={globeRef} className="w-full flex justify-center"></div>
          </div>
          <div className="bg-black bg-opacity-40 rounded-lg p-4">
            <h3 className="text-lg font-bold mb-4 flex items-center">
              <Activity className="w-5 h-5 mr-2 text-green-400" />
              Active Aircraft Monitor
            </h3>
            <div className="max-h-64 overflow-y-auto space-y-2">
              {filteredAircraft.map((a) => (
                <div
                  key={a.tailNumber}
                  onClick={() => setSelectedTailNumber(a.tailNumber)}
                  className="p-3 rounded bg-gray-800 cursor-pointer hover:bg-gray-700"
                >
                  <div className="flex justify-between">
                    <div>
                      <div className="font-bold">{a.tailNumber}</div>
                      <div className="text-xs text-gray-400">
                        {a.airline} • {a.aircraftType}
                      </div>
                    </div>
                    <div>{a.riskLevel}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* DETAILS */}
        <div className="col-span-3 space-y-4">
          {selectedAircraftData ? (
            <div className="bg-black bg-opacity-40 rounded-lg p-4">
              <h3 className="font-bold flex items-center">
                <Plane className="w-5 h-5 mr-2 text-blue-400" />
                {selectedAircraftData.tailNumber}
              </h3>
              <div className="text-sm mt-2 space-y-1">
                <div>Airline: {selectedAircraftData.airline}</div>
                <div>Type: {selectedAircraftData.aircraftType}</div>
                <div>Status: {selectedAircraftData.status}</div>
                <div>Alt: {selectedAircraftData.altitude} ft</div>
                <div>Speed: {selectedAircraftData.speed} kt</div>
                <div>Risk: {selectedAircraftData.riskLevel}</div>
              </div>
            </div>
          ) : (
            <div className="bg-black bg-opacity-40 rounded-lg p-4">
              <h3 className="font-bold flex items-center">
                <Gauge className="w-5 h-5 mr-2 text-gray-400" />
                Aircraft Details
              </h3>
              <div className="text-center text-gray-500">
                Select an aircraft
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default GuardianEyeDashboard;
