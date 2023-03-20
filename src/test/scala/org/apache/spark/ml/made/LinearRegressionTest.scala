package org.apache.spark.ml.made
import com.google.common.io.Files

import org.apache.spark.ml
import org.apache.spark.ml.linalg.{Vector, VectorUDT, Vectors}
import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.types.{StructField, StructType}
import org.scalatest.flatspec._
import org.scalatest.matchers._
import breeze.linalg._
import org.apache.spark.ml.{Pipeline, PipelineModel}

import scala.collection.JavaConverters._

class LinearRegressionTest extends AnyFlatSpec with should.Matchers with WithSpark {
  lazy val data: DataFrame = LinearRegressionTest._data

  "Model" should "predict target" in {
    // 1.5 * x_0 + 0.3 * x_1 - 0.7 = target
    val model: LinearRegressionModel = new LinearRegressionModel(
      w = Vectors.dense(1.5, 0.3).toDense,
      b = -0.7
    ).setInputCol("x")
      .setTargetCol("y")

    validateModel(model, data)
  }

  "Estimator" should "learn weights and biases" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setTargetCol("y")
      .setWeightsLR(1e-4)
      .setBiasLR(1e-1)
      .setNumEpochs(150)

    val model = estimator.fit(data)

    model.w(0) should be(1.5 +- 1e-2)
    model.w(1) should be(0.3 +- 1e-2)
    model.b should be(-0.7 +- 1e-1)
  }

  "Estimator" should "produce functional model" in {
    val estimator = new LinearRegression()
      .setInputCol("x")
      .setTargetCol("y")
      .setWeightsLR(1e-4)
      .setBiasLR(1e-1)
      .setNumEpochs(150)

    val model = estimator.fit(data)

    validateModel(model, data)
  }

  "Estimator" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("x")
        .setTargetCol("y")
        .setWeightsLR(1e-4)
        .setBiasLR(1e-1)
        .setNumEpochs(150)
    ))

    val temp_folder = Files.createTempDir()
    pipeline.write.overwrite().save(temp_folder.getAbsolutePath)

    val reRead = Pipeline.load(temp_folder.getAbsolutePath)
    val model = reRead.fit(data).stages(0).asInstanceOf[LinearRegressionModel]

    model.w(0) should be(1.5 +- 1e-2)
    model.w(1) should be(0.3 +- 1e-2)
    model.b should be(-0.7 +- 1e-1)

    validateModel(model, data)
  }

  "Model" should "work after re-read" in {
    val pipeline = new Pipeline().setStages(Array(
      new LinearRegression()
        .setInputCol("x")
        .setTargetCol("y")
        .setWeightsLR(1e-4)
        .setBiasLR(1e-1)
        .setNumEpochs(150)
    ))

    val model = pipeline.fit(data)

    val temp_folder = Files.createTempDir()
    model.write.overwrite().save(temp_folder.getAbsolutePath)

    val reRead: PipelineModel = PipelineModel.load(temp_folder.getAbsolutePath)
    val loaded_model: LinearRegressionModel = reRead.stages(0).asInstanceOf[LinearRegressionModel]

    loaded_model.w(0) should be(1.5 +- 1e-2)
    loaded_model.w(1) should be(0.3 +- 1e-2)
    loaded_model.b should be(-0.7 +- 1e-1)

    validateModel(loaded_model, data)
  }

  private def validateModel(model: LinearRegressionModel, data: DataFrame): Unit = {
    val processedData = model.transform(data)
    val vectors: Array[(Vector, Double)] = processedData.collect()
      .map(row => (row.getAs[Vector]("x"), row.getAs[Double]("y")))

    def f(x: Vector, w: Vector, b: Double) = (x.asBreeze dot w.asBreeze) + b

//    def cast(z: Vector) = z.asInstanceOf[ml.linalg.DenseVector].asBreeze(0)

    for (vector <- vectors) {
      vector match {
        case (x, z) =>
          z should be(f(x, model.w, model.b) +- 1)
      }
    }
  }
}

object LinearRegressionTest extends WithSpark {
  lazy val random_data: breeze.linalg.DenseMatrix[Double] = DenseMatrix.rand(1000, 2) * 100.0
  lazy val W: DenseVector[Double] = DenseVector(1.5, 0.3)
  lazy val b: Double = -0.7
  lazy val target: DenseVector[Double] = random_data * W + b
  lazy val data_by_row: Seq[Row] = (0 until target.length)
    .map(i => Row(Vectors.dense(random_data(i, ::).t.toArray), Vectors.dense(target(i))))

  lazy val schema: StructType = StructType(
    Array(
      StructField("x", new VectorUDT()),
      StructField("y", new VectorUDT())
    ))
  lazy val _data: DataFrame = sqlc.createDataFrame(data_by_row.asJava, schema)
}
