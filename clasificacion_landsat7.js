////// Clasificación Supervisada - Landsat 8 - RF - 2023 //////

Map.centerObject(a, 10);
Map.addLayer(a, {color: 'red'}, 'Zona de Estudio');

var maskClouds = function(image) {
  var qa        = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0)
                    .and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(cloudMask);
};

var dataset2023 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
    .filterDate('2022-12-01', '2023-03-31')
    .filterBounds(a)
    .map(maskClouds);

print('Número de imágenes disponibles 2023:', dataset2023.size());

var datasetProcessed2023 = dataset2023.map(function(image) {
  return image.multiply(0.0000275).add(-0.2).clip(a)
    .copyProperties(image, ['system:time_start']);
});

// ============================================
// DEM — ASTER 2023
// ============================================
var dem = ee.Image('projects/ee-dlorca012/assets/AsterValle')
  .rename('elevation')
  .resample('bilinear')
  .clip(a);

var slope     = ee.Terrain.slope(dem).rename('Slope');
var aspect    = ee.Terrain.aspect(dem).rename('Aspect');
var texture   = dem.reduceNeighborhood({
  reducer: ee.Reducer.stdDev(), kernel: ee.Kernel.square(3)
}).rename('Texture');
var convexity = dem.convolve(ee.Kernel.laplacian8()).rename('Convexity');

var kernelTPI = ee.Kernel.circle({radius: 250, units: 'meters'});
var meanElev  = dem.reduceNeighborhood({reducer: ee.Reducer.mean(), kernel: kernelTPI});
var tpi       = dem.subtract(meanElev).rename('TPI');
var tri       = dem.reduceNeighborhood({
  reducer: ee.Reducer.stdDev(), kernel: kernelTPI
}).rename('TRI');

var rugosidad = dem.reduceNeighborhood({
  reducer: ee.Reducer.max(), kernel: ee.Kernel.square(3)
}).subtract(
  dem.reduceNeighborhood({
    reducer: ee.Reducer.min(), kernel: ee.Kernel.square(3)
  })
).rename('Rugosidad');

var addIndices = function(image) {
  var ndvi     = image.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI');
  var ndsi     = image.normalizedDifference(['SR_B3', 'SR_B6']).rename('NDSI');
  var ndgi     = image.normalizedDifference(['SR_B3', 'SR_B4']).rename('NDGI');
  var imd      = image.normalizedDifference(['SR_B4', 'SR_B2']).rename('IMD');
  var elevBand = dem.select('elevation');
  return image.addBands([ndvi, ndsi, ndgi, imd, elevBand,
                         slope, aspect, convexity, tpi, tri,
                         texture, rugosidad]);
};

var datasetWithIndices2023 = datasetProcessed2023.map(addIndices);
var meanImage2023 = datasetWithIndices2023.median();
var meanImage2023_filled = meanImage2023.focal_mean({
  radius: 1, kernelType: 'square', units: 'pixels', iterations: 3
}).blend(meanImage2023);

print('Bandas meanImage2023:', meanImage2023_filled.bandNames());

Map.addLayer(meanImage2023_filled.select('NDVI').unmask(-9999).eq(-9999), {
  min: 0, max: 1, palette: ['white','red']
}, 'Píxeles sin datos 2023', false);

var training_data = GlaciarDescubierto.merge(GlaciarCubierto).merge(Morrena)
                    .merge(Agua).merge(AbanicoAluvial).merge(Coluvios)
                    .merge(AfloramientoDeRoca).merge(Llanura);

var training = meanImage2023_filled.sampleRegions({
  collection: training_data, properties: ['land_class'],
  scale: 30, tileScale: 4
});

print('Total puntos de entrenamiento:', training.size());

var sample      = training.randomColumn('random', 42);
var trainingSet = sample.filter(ee.Filter.lt('random', 0.7));
var testingSet  = sample.filter(ee.Filter.gte('random', 0.7));

print('Puntos entrenamiento (70%):', trainingSet.size());
print('Puntos prueba (30%):',       testingSet.size());

var classifierRF = ee.Classifier.smileRandomForest(500).train({
  features: trainingSet, classProperty: 'land_class',
  inputProperties: meanImage2023_filled.bandNames()
});

var classifiedRF = meanImage2023_filled.classify(classifierRF);
Map.addLayer(classifiedRF, {
  min: 1, max: 8,
  palette: ['#b8c2ef','#8b92b4','#40ffcc','#3665bd','#e9bb6e','#ffa500','#c2534f','#00b900']
}, 'Clasificación RF 2023');

var classifiedTestRF  = testingSet.classify(classifierRF);
var confusionMatrixRF = classifiedTestRF.errorMatrix('land_class', 'classification');
print('Matriz de confusión RF 2023:',  confusionMatrixRF);
print('Exactitud general RF 2023:',    confusionMatrixRF.accuracy());

Export.image.toDrive({
  image: classifiedRF, description: 'Clasificacion_RF_2023_ASTER',
  scale: 30, region: a, maxPixels: 1e12,
  folder: 'GEE_Exports', fileFormat: 'GeoTIFF'
});

var importanceDict     = classifierRF.explain().get('importance');
var variableImportance = ee.Dictionary(importanceDict);
var totalImportance    = variableImportance.values().reduce(ee.Reducer.sum());
var relativeImportance = variableImportance.map(function(key, value) {
  return ee.Number(value).multiply(100).divide(totalImportance);
});
print('Relative Importance RF 2023:', relativeImportance);

var chart = ui.Chart.feature.byProperty({
  features: ee.FeatureCollection([ee.Feature(null, relativeImportance)])
}).setOptions({
  title: 'Feature Importance - RF 2023',
  vAxis: {title: 'Importance (%)'}, hAxis: {title: 'Features'},
  legend: {position: 'none'}, colors: ['#1f77b4']
});
print(chart);

var clasesNombres = {
  1:'GlaciarDescubierto', 2:'GlaciarCubierto', 3:'Morrena',
  4:'Agua', 5:'AbanicoAluvial', 6:'Coluvios',
  7:'AfloramientoRoca', 8:'Llanura'
};

[1,2,3,4,5,6,7,8].forEach(function(c) {
  var training_bin = training.map(function(feature) {
    return feature.set('binaria', ee.Number(feature.get('land_class')).eq(c));
  });
  var rf_bin = ee.Classifier.smileRandomForest(500).train({
    features: training_bin, classProperty: 'binaria',
    inputProperties: meanImage2023_filled.bandNames()
  });
  print('Importancia variables - ' + clasesNombres[c],
        ee.Dictionary(rf_bin.explain().get('importance')));
});

var clasificadorEntropia2023 = ee.Classifier.smileRandomForest(800)
    .setOutputMode('MULTIPROBABILITY')
    .train({
      features: trainingSet, classProperty: 'land_class',
      inputProperties: meanImage2023_filled.bandNames()
    });

var clasificadaProb2023     = meanImage2023_filled.classify(clasificadorEntropia2023);
var etiquetasClases         = training_data.aggregate_array('land_class').distinct().sort();
var etiquetasClasesList     = etiquetasClases.map(function(id) {
  return ee.String(ee.Number(id).toInt());
});

var probasPorClase2023 = clasificadaProb2023.arrayFlatten([etiquetasClasesList]);
var entropy2023        = probasPorClase2023.multiply(probasPorClase2023.log())
                          .reduce(ee.Reducer.sum()).multiply(-1).rename('entropy');

Map.addLayer(entropy2023, {min:0, max:2, palette:['white','yellow','red']}, 'Entropía 2023');

var puntosConEntropia2023 = training_data.map(function(pt) {
  var value  = entropy2023.reduceRegion({
    reducer: ee.Reducer.first(), geometry: pt.geometry(),
    scale: 30, maxPixels: 1e13
  }).get('entropy');
  var coords = pt.geometry().coordinates();
  return pt.set({
    'entropia': value,
    'longitud': ee.List(coords).get(0),
    'latitud':  ee.List(coords).get(1)
  }).setGeometry(null);
});

Export.table.toDrive({
  collection: puntosConEntropia2023, description: 'Puntos_con_Entropia_2023',
  fileNamePrefix: 'Entropia_Puntos_2023', fileFormat: 'CSV'
});
Export.image.toDrive({
  image: entropy2023, description: 'Entropia_Raster_2023_ASTER',
  fileNamePrefix: 'Entropia_2023_ASTER', scale: 30, region: a,
  maxPixels: 1e12, folder: 'GEE_Exports', fileFormat: 'GeoTIFF'
});

var classifiedTrain2023 = trainingSet.classify(classifierRF);
var confMatrixTrain2023 = classifiedTrain2023.errorMatrix('land_class', 'classification');
print('=== COMPARACIÓN TRAIN vs TEST 2023 ===');
print('Accuracy ENTRENAMIENTO:', confMatrixTrain2023.accuracy());
print('Accuracy PRUEBA:',        confusionMatrixRF.accuracy());
print('Kappa ENTRENAMIENTO:',    confMatrixTrain2023.kappa());
print('Kappa PRUEBA:',           confusionMatrixRF.kappa());

print('=== MÉTRICAS POR CLASE 2023 ===');
print('Producer Accuracy:', confusionMatrixRF.producersAccuracy());
print('Consumer Accuracy:', confusionMatrixRF.consumersAccuracy());
print('Kappa general:',     confusionMatrixRF.kappa());
print('Matriz completa:',   confusionMatrixRF);

var clases   = [1,2,3,4,5,6,7,8];
print('Conteo de puntos por clase 2023:', ee.FeatureCollection(clases.map(function(c) {
  return ee.Feature(null, {
    'clase': c,
    'n_puntos': training_data.filter(ee.Filter.eq('land_class', c)).size()
  });
})));

var puntosConElevacion2023 = training_data.map(function(pt) {
  var elev   = dem.reduceRegion({
    reducer: ee.Reducer.first(), geometry: pt.geometry(), scale: 30
  }).get('elevation');
  var coords = pt.geometry().coordinates();
  return pt.set({
    'elevation': elev,
    'lon': ee.List(coords).get(0),
    'lat': ee.List(coords).get(1)
  }).setGeometry(null);
});

Export.table.toDrive({
  collection: puntosConElevacion2023, description: 'Training_Points_Elevation_2023',
  fileNamePrefix: 'Training_Points_Elevation_ASTER2023', fileFormat: 'CSV'
});

var trainingConFolds2023 = training.randomColumn('fold_raw', 42)
  .map(function(f) {
    var fold = ee.Number(f.get('fold_raw')).multiply(5).floor().min(4);
    return f.set('fold', fold);
  });

var kFoldResultados2023 = ee.List.sequence(0, 4).map(function(k) {
  var testFold  = trainingConFolds2023.filter(ee.Filter.eq('fold', k));
  var trainFold = trainingConFolds2023.filter(ee.Filter.neq('fold', k));
  var rfFold    = ee.Classifier.smileRandomForest(500).train({
    features: trainFold, classProperty: 'land_class',
    inputProperties: meanImage2023_filled.bandNames()
  });
  var matriz = testFold.classify(rfFold).errorMatrix('land_class', 'classification');
  return ee.Feature(null, {
    'fold': k, 'accuracy': matriz.accuracy(), 'kappa': matriz.kappa(),
    'n_train': trainFold.size(), 'n_test': testFold.size()
  });
});

var kFoldFC2023 = ee.FeatureCollection(kFoldResultados2023);
print('=== RESULTADOS K-FOLD 2023 (k=5) ===', kFoldFC2023);
print('Accuracy media K-Fold 2023:', kFoldFC2023.aggregate_mean('accuracy'));
print('Accuracy std K-Fold 2023:',   kFoldFC2023.aggregate_total_sd('accuracy'));
print('Kappa media K-Fold 2023:',    kFoldFC2023.aggregate_mean('kappa'));
print('Kappa std K-Fold 2023:',      kFoldFC2023.aggregate_total_sd('kappa'));

Export.table.toDrive({
  collection: kFoldFC2023, description: 'KFold_5_Resultados_2023',
  fileNamePrefix: 'KFold_5_Resultados_ASTER2023', fileFormat: 'CSV'
});

var imagenClaseEntropia2023 = classifiedRF.rename('clase').addBands(entropy2023);
var muestrasEntropia2023    = imagenClaseEntropia2023.stratifiedSample({
  numPoints: 200, classBand: 'clase', region: a,
  scale: 30, seed: 42, geometries: false
});

var statsEntropia2023 = clases.map(function(c) {
  var muestra = muestrasEntropia2023.filter(ee.Filter.eq('clase', c));
  var stats   = muestra.reduceColumns({
    reducer: ee.Reducer.mean()
      .combine(ee.Reducer.stdDev(), null, true)
      .combine(ee.Reducer.percentile([25,50,75]), null, true)
      .combine(ee.Reducer.min(), null, true)
      .combine(ee.Reducer.max(), null, true),
    selectors: ['entropy']
  });
  return ee.Feature(null, {
    'clase': c, 'media': stats.get('mean'), 'std': stats.get('stdDev'),
    'p25': stats.get('p25'), 'mediana': stats.get('p50'),
    'p75': stats.get('p75'), 'min': stats.get('min'), 'max': stats.get('max')
  });
});

var statsFC2023 = ee.FeatureCollection(statsEntropia2023);
print('=== ESTADÍSTICAS ENTROPÍA POR CLASE 2023 ===', statsFC2023);

Export.table.toDrive({
  collection: muestrasEntropia2023, description: 'Entropia_por_Clase_2023',
  fileNamePrefix: 'Entropia_por_Clase_ASTER2023', fileFormat: 'CSV'
});
Export.table.toDrive({
  collection: statsFC2023, description: 'Stats_Entropia_por_Clase_2023',
  fileNamePrefix: 'Stats_Entropia_por_Clase_ASTER2023', fileFormat: 'CSV'
});

var areaImage = ee.Image.pixelArea().addBands(classifiedRF.rename('classification'));
var areas     = areaImage.reduceRegion({
  reducer: ee.Reducer.sum().group({groupField: 1, groupName: 'class'}),
  geometry: a, scale: 30, maxPixels: 1e13
});
var areaHectares = ee.List(areas.get('groups')).map(function(feature) {
  var dict = ee.Dictionary(feature);
  return dict.set('area_ha', ee.Number(dict.get('sum')).divide(10000));
});
print('Área por clase en hectáreas (RF 2023):', areaHectares);

var landsat_filtro_2023 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
  .filterBounds(a).filterDate('2022-12-01', '2023-03-31')
  .filterMetadata('CLOUD_COVER', 'less_than', 5).map(maskClouds);

print('Número de imágenes filtradas 2023:', landsat_filtro_2023.size());

var imagen_mediana_2023 = landsat_filtro_2023
  .map(function(image) {
    return image.multiply(0.0000275).add(-0.2).clip(a);
  }).median().multiply(10000).toUint16();

Map.addLayer(imagen_mediana_2023, {
  bands: ['SR_B4','SR_B3','SR_B2'], min: 500, max: 3000, gamma: 1.4
}, 'Landsat 8 - Color Natural 2023');

Export.image.toDrive({
  image: imagen_mediana_2023, description: 'Landsat_8_Color_Natural_2023',
  folder: 'GEE_Exports', region: a, scale: 30,
  crs: 'EPSG:32719', fileFormat: 'GeoTIFF'
});

// ============================================
// EXPORTAR IMPORTANCIA POR CLASE EN CSV
// ============================================

var clasesIds = [1,2,3,4,5,6,7,8];
var clasesNombresExport = {
  1:'GlaciarDescubierto', 2:'GlaciarCubierto', 3:'Morrena',
  4:'Agua', 5:'AbanicoAluvial', 6:'Coluvios',
  7:'AfloramientoRoca', 8:'Llanura'
};

var importanciasPorClase = ee.FeatureCollection(
  clasesIds.map(function(c) {
    var training_bin = training.map(function(feature) {
      return feature.set('binaria', ee.Number(feature.get('land_class')).eq(c));
    });

    var rf_bin = ee.Classifier.smileRandomForest(100).train({
      features:        training_bin,
      classProperty:   'binaria',
      inputProperties: meanImage2023_filled.bandNames()
    });

    var imp = ee.Dictionary(rf_bin.explain().get('importance'));

    // Agregar clase y nombre como propiedades
    return ee.Feature(null, imp
      .set('clase_id',     c)
      .set('clase_nombre', clasesNombresExport[c])
    );
  })
);

print('Importancia por clase (tabla):', importanciasPorClase);

Export.table.toDrive({
  collection:     importanciasPorClase,
  description:    'Importancia_Variables_por_Clase_2023',
  fileNamePrefix: 'Importancia_por_Clase_2023',
  fileFormat:     'CSV'
});
