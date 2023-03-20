package org.apache.spark.ml.made

import breeze.linalg.InjectNumericOps
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.linalg.{DenseVector, Vector, VectorUDT, Vectors}
import org.apache.spark.ml.param.{DoubleParam, IntParam, Param, ParamMap, Params}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, Encoder}

trait HasTargetCol extends Params {
  final val targetCol: Param[String] = new Param[String](this, "targetCol", "target column name")
  final def getTargetCol: String = $(targetCol)
  def setTargetCol(value: String): this.type = set(targetCol, value)
}

trait LinearRegressionParams extends HasInputCol with HasTargetCol with HasOutputCol {
  def setInputCol(value: String): this.type = set(inputCol, value)

  val weightsLR = new DoubleParam(this, "weightsLR", "Learning rate for weights")
  def setWeightsLR(value: Double): this.type = set(weightsLR, value)
  setDefault(weightsLR -> 1e-4)

  val biasLR = new DoubleParam(this, "biasLR", "Learning rate for bias")
  def setBiasLR(value: Double): this.type = set(biasLR, value)
  setDefault(biasLR -> 1e-1)

  val numEpochs = new IntParam(this, "numEpochs", "Number of training epochs")
  def setNumEpochs(value: Int): this.type = set(numEpochs, value)
  setDefault(numEpochs -> 150)

  protected def validateAndTransformSchema(schema: StructType): StructType = {
    SchemaUtils.checkColumnType(schema, getInputCol, new VectorUDT())
    SchemaUtils.checkColumnType(schema, getTargetCol, new VectorUDT())
    schema
  }
}


class LinearRegression(override val uid: String) extends Estimator[LinearRegressionModel] with LinearRegressionParams
  with DefaultParamsWritable {

  def this() = this(Identifiable.randomUID("linearRegression"))

  override def fit(dataset: Dataset[_]): LinearRegressionModel = {

    implicit val encoder: Encoder[Vector] = ExpressionEncoder()

    val vectors: Dataset[(Vector, Vector)] = dataset.select(dataset($(inputCol)).as[Vector], dataset($(targetCol)).as[Vector])

    val dim: Int = AttributeGroup.fromStructField(dataset.schema($(inputCol))).numAttributes.getOrElse(
      vectors.first()._1.size
    )

    var w: breeze.linalg.Vector[Double] = breeze.linalg.DenseVector.fill(dim) {
      1
    }
    var b = 1.0

    val N = dataset.count()
    val num_epochs = $(numEpochs)

    for (_ <- 0 until num_epochs) {
      val (eps, eps_x) =
        vectors.rdd
          .map {
            case (x, y) => (x.asBreeze, y.asBreeze)
          }
          .map({ case (x, y) =>
            val eps = (x dot w) + b - y(0)
            (eps, x * eps)
          }).reduce({ case ((eps1, eps_x1), (eps2, eps_x2)) => (eps1 + eps2, eps_x1 + eps_x2) })


      w = w - $(weightsLR) / N * eps_x
      b = b - $(biasLR) / N * eps
    }

    copyValues(new LinearRegressionModel(
      Vectors.fromBreeze(w), b
    )).setParent(this)
  }

  override def copy(extra: ParamMap): Estimator[LinearRegressionModel] = defaultCopy(extra)

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

}

object LinearRegression extends DefaultParamsReadable[LinearRegression]


class LinearRegressionModel private[made](
                                           override val uid: String,
                                           val w: DenseVector,
                                           val b: Double) extends Model[LinearRegressionModel] with LinearRegressionParams with MLWritable {


  private[made] def this(w: Vector, b: Double) =
    this(Identifiable.randomUID("linearRegressionModel"), w.toDense, b)

  override def copy(extra: ParamMap): LinearRegressionModel = copyValues(
    new LinearRegressionModel(w, b), extra)

  override def transform(dataset: Dataset[_]): DataFrame = {
    val transformUdf =
      dataset.sqlContext.udf.register(uid + "_transform",
        (x: Vector) => {
          (x.asBreeze dot w.asBreeze) + b
        })

    dataset.withColumn($(targetCol), transformUdf(dataset($(inputCol))))
  }

  override def transformSchema(schema: StructType): StructType = validateAndTransformSchema(schema)

  override def write: MLWriter = new DefaultParamsWriter(this) {
    override protected def saveImpl(path: String): Unit = {
      super.saveImpl(path)

      val vectors = w.asInstanceOf[Vector] -> b

      sqlContext.createDataFrame(Seq(vectors)).write.parquet(path + "/vectors")
    }
  }
}


object LinearRegressionModel extends MLReadable[LinearRegressionModel] {
  private implicit val vectorEncoder : Encoder[Vector] = ExpressionEncoder()
  private implicit val doubleEncoder : Encoder[Double] = ExpressionEncoder()

  override def read: MLReader[LinearRegressionModel] = new MLReader[LinearRegressionModel] {
    override def load(path: String): LinearRegressionModel = {
      val metadata = DefaultParamsReader.loadMetadata(path, sc)

      val vectors = sqlContext.read.parquet(path + "/vectors")

      val (w, b) = vectors.select(vectors("_1").as[Vector], vectors("_2").as[Double]).first()

      val model = new LinearRegressionModel(w, b)
      metadata.getAndSetParams(model)
      model
    }
  }
}