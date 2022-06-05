package banking;

/**
 *a.创建 SavingAccount 类，该类继承 Account 类
 *b.该类必须包含一个类型为 double 的 interestRate 属性
 *c.该类必须包括带有两个参数（balance 和 interest_rate）的公有构造器。
 *  该构造器必须通过调用 super(balance)将 balance 参数传递给父类构造器
 *
 */
public class SavingAccount extends Account {
    private double interestRate;

    public SavingAccount(double init_balance, double interestRate) {
        super(init_balance);
        this.interestRate = interestRate;
    }
}
