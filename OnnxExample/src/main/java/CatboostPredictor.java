
import ai.catboost.CatBoostError;
import ai.catboost.CatBoostModel;
import ai.catboost.CatBoostPredictions;

import java.util.Arrays;

public class CatboostPredictor {
    private final CatBoostModel model;

    public CatboostPredictor(String path) throws CatBoostError {
        this.model = CatBoostModel.loadModel(path);

    }

    public double[] predictProba(float[] inp) throws CatBoostError {
        CatBoostPredictions prediction = model.predict(inp, new String[]{});
        return softmax(prediction.copyObjectPredictions(0));
    }

    public static double[] softmax(final double[] x) {
        final double max = Arrays.stream(x).max().orElseThrow(() -> new IllegalArgumentException("Can't find max"));
        double sum = 0;
        final double[] exps = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            exps[i] = Math.exp(x[i] - max);
            sum += exps[i];
        }
        for (int i = 0; i < x.length; i++) {
            exps[i] /= sum;
        }
        return exps;
    }
}
