const express = require('express');
const router = express.Router();
const loginController = require('../controllers/loginController');
const registerController = require('../controllers/registerController');
const reportsController = require('../controllers/reportsController');
const addController = require('../controllers/addController');
const ForecastingController = require('../controllers/ForecastingController');
const { isAuthenticated } = require('./auth');
const multer = require('multer');


const forecastController = require('../controllers/forecastController')


const dbConfig = require('./db_config');
const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    // Specify the directory where uploaded files will be stored
    cb(null, 'uploads/');
  },
  filename: function (req, file, cb) {
    // Generate a unique filename for the uploaded file
    cb(null, Date.now() + '-' + file.originalname);
  }
});

// Create an instance of the multer middleware with the configured storage options
const upload = multer({ storage: storage });

router.get(['/', '/login'], loginController.renderLoginPage);
router.get('/register', registerController.renderRegisterPage);
router.get('/reports', isAuthenticated, reportsController.renderReportsPage);
router.get('/add', isAuthenticated, addController.renderaddPage);
router.get('/forecasting', isAuthenticated, ForecastingController.renderforecastingPage);
router.post('/register', registerController.registerUser);

router.post('/stock', addController.addStock);
router.post('/sales', isAuthenticated, addController.createSalesOrder);
router.get('/forecasts/:product_name', forecastController.forecast)
router.get('/report/:product_name', reportsController.graphdata);
router.get('/salesdata', reportsController.getSalesData);
router.post('/upload',upload.single('csvFile'),addController.uploaded);


router.get('/sales', (req, res) => {
    // Create a MySQL connection using the dbConfig.createConnection function
    const connection = dbConfig.createConnection();
  
    // Query the database to fetch product and sales order data
    const query = `
      SELECT p.product_name, p.price, p.available_quantity, s.Date, s.quantity, s.temperature, s.competitorPrice, s.competitorCount, s.advertisingCost, s.promotion
      FROM product p
      INNER JOIN salesorder s ON p.product_id = s.product_id
    `;
    connection.query(query, (error, data) => {
      if (error) {
        // Handle database query error
        console.error('Error retrieving data:', error);
        res.status(500).json({ error: 'Internal Server Error' });
      } else {
        // Send the JSON response with product and sales order data
        console.log('Data retrieved successfully:', data);
        res.json(data);
      }
  
      // Close the MySQL connection
      connection.end();
    });
  });
  

module.exports = router;
