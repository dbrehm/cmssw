import FWCore.ParameterSet.Config as cms

process = cms.Process("G4PrintGeometry")

#process.load('Configuration.Geometry.GeometryIdeal_cff')
#process.load('Configuration.Geometry.GeometryExtended_cff')
#process.load('Configuration.Geometry.GeometryExtended2015_cff')
#process.load('Configuration.Geometry.GeometryExtended2017_cff')
#process.load('Configuration.Geometry.GeometryExtended2019_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D12_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D13_cff')
#process.load('Configuration.Geometry.GeometryExtended2023D21_cff')
process.load('Configuration.Geometry.GeometryExtended2023D28_cff')

process.load('FWCore.MessageService.MessageLogger_cfi')

if hasattr(process,'MessageLogger'):
    process.MessageLogger.categories.append('G4cout')
    process.MessageLogger.categories.append('G4cerr')

from SimG4Core.PrintGeomInfo.g4PrintGeomSummary_cfi import *

process = printGeomSummary(process)
